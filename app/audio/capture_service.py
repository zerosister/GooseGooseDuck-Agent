import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from utils.logger import log_error, log_event

try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None

try:
    from importlib.metadata import version as pkg_version
except ImportError:  # pragma: no cover
    pkg_version = None


def _soundcard_version() -> Optional[str]:
    """
    检查 soundcard 版本
    """
    if pkg_version is None:
        return None
    try:
        return pkg_version("soundcard")
    except Exception:
        return None


@dataclass
class AudioChunk:
    source: str
    pcm16: np.ndarray
    sample_rate: int


def _get_loopback_microphone():
    """Windows: loopback is a Microphone with isloopback=True, same WASAPI id as default speaker."""
    if sc is None:
        return None
    try:
        speaker = sc.default_speaker()
        if speaker is None:
            return None
        for mic in sc.all_microphones(include_loopback=True):
            if getattr(mic, "isloopback", False) and mic.id == speaker.id:
                return mic
        for mic in sc.all_microphones(include_loopback=True):
            if getattr(mic, "isloopback", False):
                return mic
    except Exception:
        return None
    return None


class BackendAudioCaptureService:
    """Capture microphone + speaker loopback in two worker threads."""

    def __init__(
        self,
        logger: logging.Logger,
        session_id: str,
        sample_rate: int = 16000,
        chunk_seconds: float = 4.0,
    ) -> None:
        self._logger = logger
        self._session_id = session_id
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self._queue: "queue.Queue[AudioChunk]" = queue.Queue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._chunk_seq = 0                            # 音频块序列号
        self._mic_error_count = 0
        self._mic_last_error_log = 0.0
        self._loopback_error_count = 0
        self._loopback_last_error_log = 0.0

    @property
    def output_queue(self) -> "queue.Queue[AudioChunk]":
        return self._queue

    @property
    def is_available(self) -> bool:
        return sc is not None

    def _emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        log_event(self._logger, event_type, self._session_id, payload=payload or {})

    def _throttled_log_error(
        self,
        event_type: str,
        error: str,
        payload: Optional[Dict[str, Any]],
        *,
        counter_attr: str,
        time_attr: str,
        interval_sec: float = 5.0,
        max_full_trace: int = 2,
    ) -> None:
        count = getattr(self, counter_attr) + 1
        setattr(self, counter_attr, count)
        now = time.monotonic()
        last = getattr(self, time_attr)
        should_emit = count <= max_full_trace or (now - last) >= interval_sec
        if not should_emit:
            return
        setattr(self, time_attr, now)
        extra = dict(payload or {})
        extra["repeat_count"] = count
        log_error(self._logger, event_type, self._session_id, error, payload=extra)

    def _probe_devices(self) -> Dict[str, Any]:
        """
        探测当前系统的音频设备状态（麦克风数量、扬声器名称等），并返回一个字典用于启动时的诊断日志
        """
        ver = _soundcard_version()
        out: Dict[str, Any] = {
            "soundcard_installed": sc is not None,
            "soundcard_version": ver,
            "sample_rate": self.sample_rate,
            "chunk_seconds": self.chunk_seconds,
        }
        if sc is None:
            return out
        try:
            mic = sc.default_microphone()
            spk = sc.default_speaker()
            out["default_microphone"] = str(mic) if mic else None
            out["default_speaker"] = str(spk) if spk else None
            lb = _get_loopback_microphone()
            out["loopback_device"] = str(lb) if lb else None
            try:
                mics = sc.all_microphones()
                spks = sc.all_speakers()
                all_m = sc.all_microphones(include_loopback=True)
                out["microphone_count"] = len(mics)
                out["speaker_count"] = len(spks)
                out["loopback_candidate_count"] = sum(1 for m in all_m if getattr(m, "isloopback", False))
            except Exception as exc:
                out["enumerate_error"] = str(exc)
        except Exception as exc:
            out["probe_error"] = str(exc)
        return out

    def start(self) -> None:
        """
        启动两个守护线程进行监听
        - microphone（麦克风声音）
        - loopback（系统声音）  
        """
        if self._threads:
            return
        self._emit("backend_audio_device_probe", self._probe_devices())
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._capture_microphone_loop, name="audio-mic", daemon=True),
            threading.Thread(target=self._capture_loopback_loop, name="audio-loopback", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._emit(
            "backend_audio_threads_started",
            {"thread_names": [t.name for t in self._threads]},
        )

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        alive = [t.name for t in self._threads if t.is_alive()]
        self._emit(
            "backend_audio_threads_stopped",
            {"still_alive_after_join": alive, "queue_size": self._queue.qsize()},
        )
        self._threads = []

    def _push_audio(self, source: str, data: np.ndarray) -> None:
        """
        预处理音频数据，并推入队列
        """
        if data.size == 0:
            return
        if data.ndim > 1:
            data = data.mean(axis=1)
        self._emit("backend_audio_data_range", 
                   payload={
                       "source": source, 
                       "data_range": {
                           "min": float(np.min(data)),  # numpy标量非python自带float标量
                           "max": float(np.max(data))
                        }
                    })
        
        pcm = np.clip(data, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        self._chunk_seq += 1
        seq = self._chunk_seq
        peak = float(np.max(np.abs(pcm16))) / 32767.0 if pcm16.size else 0.0
        chunk = AudioChunk(source=source, pcm16=pcm16, sample_rate=self.sample_rate)
        self._queue.put(chunk)
        qsz = 0
        try:
            qsz = self._queue.qsize()
        except NotImplementedError:
            qsz = -1
        self._emit(
            "backend_audio_chunk_enqueued",
            {
                "seq": seq,
                "source": source,
                "pcm_bytes": int(pcm16.nbytes),
                "samples": int(pcm16.size),
                "peak_normalized": round(peak, 6),
                "queue_size_after": qsz,
            },
        )

    def _capture_microphone_loop(self) -> None:
        if sc is None:
            return
        mic = sc.default_microphone()
        if mic is None:
            log_error(
                self._logger,
                "backend_audio_mic_unavailable",
                self._session_id,
                "default_microphone() returned None",
            )
            return
        frames = int(self.sample_rate * self.chunk_seconds)
        self._emit("backend_audio_mic_loop_enter", {"device": str(mic), "frames_per_read": frames})
        try:
            with mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while not self._stop_event.is_set():
                    try:
                        data = recorder.record(numframes=frames)
                        self._push_audio("microphone", data)
                    except Exception as exc:
                        self._throttled_log_error(
                            "backend_audio_mic_record_error",
                            str(exc),
                            payload={"traceback": traceback.format_exc()},
                            counter_attr="_mic_error_count",
                            time_attr="_mic_last_error_log",
                        )
                        time.sleep(0.2)
        except Exception as exc:
            log_error(
                self._logger,
                "backend_audio_mic_loop_fatal",
                self._session_id,
                str(exc),
                payload={"traceback": traceback.format_exc()},
            )

    def _capture_loopback_loop(self) -> None:
        """Windows SoundCard: use Loopback Microphone (same id as speaker), not Speaker.recorder."""
        if sc is None:
            return
        loopback_mic = _get_loopback_microphone()
        if loopback_mic is None:
            log_error(
                self._logger,
                "backend_audio_loopback_unavailable",
                self._session_id,
                "no loopback microphone found (include_loopback=True)",
            )
            return
        frames = int(self.sample_rate * self.chunk_seconds)
        ch = min(2, max(1, int(loopback_mic.channels)))
        self._emit(
            "backend_audio_loopback_loop_enter",
            {"device": str(loopback_mic), "frames_per_read": frames, "channels": ch},
        )
        try:
            with loopback_mic.recorder(samplerate=self.sample_rate, channels=ch) as recorder:
                while not self._stop_event.is_set():
                    try:
                        data = recorder.record(numframes=frames)
                        self._push_audio("loopback", data)
                    except Exception as exc:
                        self._throttled_log_error(
                            "backend_audio_loopback_record_error",
                            str(exc),
                            payload={"traceback": traceback.format_exc()},
                            counter_attr="_loopback_error_count",
                            time_attr="_loopback_last_error_log",
                        )
                        time.sleep(0.2)
        except Exception as exc:
            log_error(
                self._logger,
                "backend_audio_loopback_loop_fatal",
                self._session_id,
                str(exc),
                payload={"traceback": traceback.format_exc()},
            )
