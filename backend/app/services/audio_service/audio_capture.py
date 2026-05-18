from .audio_recorder import AudioRecorder
from .speaker_engine import SpeakerEngine
from .asr_engine import ASREngine
import sherpa_onnx
from backend.utils.logger import log
from backend.utils.config_loader import config
from backend.utils.pathtool import get_abs_path
from backend.app.core.ggd_coordinator import GGDCoordinator
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class AudioCaptureService:
    def __init__(self, coordinator: GGDCoordinator):
        self.recorder = AudioRecorder()
        model_base = get_abs_path("backend/models")
        self.speaker_engine = SpeakerEngine(str(model_base / config.audio.spk_model))
        asr_dir = model_base / config.asr.asr_dir
        config.asr.tokens = str(asr_dir / config.asr.tokens)
        config.asr.asr_model = str(asr_dir / config.asr.asr_model)
        config.asr.graph = str(asr_dir / config.asr.graph)
        config.asr.encoder = str(asr_dir / config.asr.encoder)
        config.asr.decoder = str(asr_dir / config.asr.decoder)
        config.asr.joiner = str(asr_dir / config.asr.joiner)
        self.asr_engine = ASREngine(args=config.asr)
        self.coordinator = coordinator
        
        # 初始化 VAD
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(model_base / config.audio.vad_model)
        vad_config.sample_rate = 16000
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        log.info(f"vad init success, model: {vad_config.silero_vad.model}")

        # 初始化产品队列
        self.audio_queue = queue.Queue(maxsize=100) # 缓冲区
        self.running = False

        # 初始化线程池
        self.segment_executor = ThreadPoolExecutor(max_workers=2)
        self.asr_executor = ThreadPoolExecutor(max_workers=1)
        self.speaker_executor = ThreadPoolExecutor(max_workers=1)

        # 初始化 ASR/speaker识别结果 信号量
        self.segment_semaphore = threading.Semaphore(4)
        self.result_lock = threading.Lock()
    
    def _process_segment_async(self, samples, precise_start_time, duration):
        try:
            asr_future = self.asr_executor.submit(self.asr_engine.transcribe, samples)
            speaker_future = self.speaker_executor.submit(self.speaker_engine.identify, samples)

            text = asr_future.result()
            spk_name = speaker_future.result()

            log.debug(
                f"audio 说话人：{spk_name}，识别结果：{text}，"
                f"物理时间戳：{precise_start_time}，持续时间：{duration}"
            )

            if self.coordinator:
                with self.result_lock:
                    self.coordinator.on_audio_result(text, precise_start_time, duration, spk_name)

        except Exception as e:
            log.exception(f"音频片段处理失败: {e}")
        finally:
            self.segment_semaphore.release()
    
    def _processing_worker(self):
        """消费者线程：专门负责 VAD、ASR 和 Speaker ID"""
        while self.running:
            try:
                # 设置超时避免死锁
                samples = self.audio_queue.get(timeout=1)
                self.vad.accept_waveform(samples)
                
                while not self.vad.empty():
                    segment = self.vad.front
                    if len(segment.samples) < 0.5 * 16000:
                        self.vad.pop()
                        continue
                    
                    # 计算物理时间戳：
                    # 流开始时间 + (该段在音频流中的起始采样点 / 采样率)
                    precise_start_time = self.stream_start_wall_time + (segment.start / 16000.0)
                    duration = len(segment.samples) / 16000.0
                    
                    # 复制一份 samples
                    samples = np.copy(segment.samples)

                    if self.segment_semaphore.acquire(blocking=False):
                        self.segment_executor.submit(
                            self._process_segment_async,
                            samples,
                            precise_start_time,
                            duration,
                        )
                    else:
                        log.warning("音频片段处理任务积压，丢弃当前 segment")

                    self.vad.pop()
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def run(self):
        self.running = True
        log.info("系统监听已启动...")
                
        # 启动处理线程
        worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        worker_thread.start()

        # 记录录音流开始的时间戳 
        self.stream_start_wall_time = time.time()
        
        # 主线程仅负责采集（生产者）
        try:
            for samples, timestamp in self.recorder.record_generator():
                if not self.running:
                    break
                try:
                    # 如果队列满了，说明处理速度确实跟不上，这里会报错或丢弃
                    self.audio_queue.put_nowait(samples)
                except queue.Full:
                    log.warning("处理队列已满，可能出现数据丢失")
        except Exception as e:
            log.error(f"采集异常: {e}")

    def stop(self):
        self.recorder.stop()