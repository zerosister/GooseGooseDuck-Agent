import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wave

import numpy as np

from PIL import Image

from utils.logger import LOG_DIR, log_error, log_event

from rapidocr_onnxruntime import RapidOCR
from funasr import AutoModel
import os


class InputProcessor:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.ocr_engine = RapidOCR() if RapidOCR else None
        self.asr_model = None
        if AutoModel:
            try:
                self.asr_model = AutoModel(model="paraformer-zh")
                log_event(self.logger, "asr_model_loaded", "asr_model_loaded", payload={"model": "paraformer-zh"})
            except Exception as exc:
                log_error(self.logger, "asr_model_load_exception", "asr_model_load_exception", payload={"model": "paraformer-zh", "error": str(exc)})
                self.asr_model = None
                raise exc

    @staticmethod
    def _seat_rois(width: int, height: int) -> Dict[int, Tuple[int, int, int, int]]:
        # 3x5 布局的近似 ROI（x1, y1, x2, y2），按分辨率缩放
        rois: Dict[int, Tuple[int, int, int, int]] = {}
        idx = 1
        x_start, x_gap = 0.04, 0.19
        y_start, y_gap = 0.08, 0.145
        roi_w, roi_h = 0.16, 0.115
        for r in range(5):
            for c in range(3):
                x1 = int((x_start + c * x_gap) * width)
                y1 = int((y_start + r * y_gap) * height)
                x2 = int(min(width, x1 + int(roi_w * width)))
                y2 = int(min(height, y1 + int(roi_h * height)))
                rois[idx] = (x1, y1, x2, y2)
                idx += 1
        return rois

    @staticmethod
    def _to_hsv_channels(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        h = np.zeros_like(cmax)
        s = np.zeros_like(cmax)
        v = cmax
        mask = delta != 0
        r_mask = (cmax == r) & mask
        g_mask = (cmax == g) & mask
        b_mask = (cmax == b) & mask
        h[r_mask] = (60.0 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360.0) % 360.0
        h[g_mask] = 60.0 * ((b[g_mask] - r[g_mask]) / delta[g_mask] + 2.0)
        h[b_mask] = 60.0 * ((r[b_mask] - g[b_mask]) / delta[b_mask] + 4.0)
        nz = cmax != 0
        s[nz] = delta[nz] / cmax[nz]
        return h, s, v

    @staticmethod
    def _border_mask(h: int, w: int, thickness: int = 3) -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        if h <= thickness * 2 or w <= thickness * 2:
            mask[:] = True
            return mask
        mask[:thickness, :] = True
        mask[-thickness:, :] = True
        mask[:, :thickness] = True
        mask[:, -thickness:] = True
        return mask

    def process_frame(self, session_id: str, frame_b64: str) -> Dict:
        """
        处理图像帧
        """
        try:
            image_bytes = base64.b64decode(frame_b64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            frame_path = LOG_DIR / f"{session_id}_{ts}.png"
            image.save(frame_path)

            text = ""
            if self.ocr_engine:
                result, _ = self.ocr_engine(str(frame_path))
                lines = []
                for item in result or []:
                    if len(item) >= 2 and item[1]:
                        lines.append(str(item[1]))
                text = " ".join(lines).strip()

            payload = {"frame_path": str(frame_path), "ocr_text": text}
            log_event(self.logger, "ocr_result", session_id, payload=payload)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "text": text,
                "source": "ocr",
                "frame_path": str(frame_path),
            }
        except Exception as exc:
            log_error(self.logger, "ocr_exception", session_id, str(exc))
            return {"timestamp": datetime.utcnow().isoformat(), "text": "", "source": "ocr", "frame_path": ""}

    def initialize_seat_map_from_frame(self, session_id: str, frame_path: str) -> Dict[int, str]:
        """
        初始化座位表
        """
        seat_map: Dict[int, str] = {}
        if not frame_path:
            log_error(self.logger, "seat_map_init_exception", session_id, "frame_path is None", payload={"frame_path": frame_path})
            return seat_map
        try:
            image = Image.open(frame_path).convert("RGB")
            width, height = image.size
            rois = self._seat_rois(width, height)
            seat_dir = LOG_DIR / f"seat_crops_{session_id}"
            os.makedirs(seat_dir, exist_ok=True)
            for seat_num, (x1, y1, x2, y2) in rois.items():
                name = f"Unknown_{seat_num:02d}"
                # 先裁剪出座位区域，然后进行OCR识别
                if self.ocr_engine:
                    crop = image.crop((x1, y1, x2, y2))
                    crop_path = seat_dir / f"seat_crop_{session_id}_{seat_num:02d}.png"
                    crop.save(crop_path)
                    result, _ = self.ocr_engine(str(crop_path))
                    parts: List[str] = []
                    for item in result or []:
                        if len(item) >= 2 and item[1]:
                            parts.append(str(item[1]).strip())
                    merged = "".join(parts).strip()
                    if merged:
                        name = merged
                seat_map[seat_num] = name
            log_event(self.logger, "seat_map_init_success", session_id, payload={"seat_map": seat_map})
        except Exception as exc:
            log_error(self.logger, "seat_map_init_exception", session_id, str(exc), payload={"frame_path": frame_path})
            if not seat_map:
                seat_map = {i: f"Unknown_{i:02d}" for i in range(1, 16)}
        return seat_map

    def detect_speaker_status_from_frame(self, session_id: str, frame_path: str) -> Dict[str, object]:
        """
        hsv 检测边框是否有发言黄色边框
        """
        if not frame_path:
            log_error(self.logger, "speaker_detect_exception", session_id, "frame_path is None", payload={"frame_path": frame_path})
            return {"speaker_num": None, "scores": {}}
        scores: Dict[int, float] = {}
        try:
            image = Image.open(frame_path).convert("RGB")
            rgb = np.array(image)
            h, s, v = self._to_hsv_channels(rgb)
            rois = self._seat_rois(rgb.shape[1], rgb.shape[0])
            for seat_num, (x1, y1, x2, y2) in rois.items():
                roi_h = h[y1:y2, x1:x2]
                roi_s = s[y1:y2, x1:x2]
                roi_v = v[y1:y2, x1:x2]
                if roi_h.size == 0:
                    scores[seat_num] = 0.0
                    continue
                border = self._border_mask(roi_h.shape[0], roi_h.shape[1], thickness=3)
                yellow = (roi_h >= 35.0) & (roi_h <= 70.0) & (roi_s >= 0.45) & (roi_v >= 0.55)
                score = float(np.mean(yellow[border])) if np.any(border) else 0.0
                scores[seat_num] = score
            if not scores:
                return {"speaker_num": None, "scores": {}}
            max_seat = max(scores, key=scores.get)
            max_score = scores[max_seat]
            speaker_num = max_seat if max_score >= 0.08 else None
            log_event(self.logger, "speaker_detect_success", session_id, payload={"speaker_num": speaker_num, "scores": scores})
            return {"speaker_num": speaker_num, "scores": scores}
        except Exception as exc:
            log_error(self.logger, "speaker_detect_exception", session_id, str(exc), payload={"frame_path": frame_path})
            return {"speaker_num": None, "scores": {}}

    def slice_audio_by_event(
        self,
        session_id: str,
        event: Dict[str, object],
        audio_ring: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """
        根据发言事件切片音频
        """
        start_ts = float(event.get("start_ts", 0.0) or 0.0)
        end_ts = float(event.get("end_ts", 0.0) or 0.0)
        player_id = str(event.get("speaker_name", "Unknown"))
        if end_ts <= start_ts:
            return {"player_id": player_id, "audio_files": [], "start_ts": start_ts, "end_ts": end_ts}

        selected: List[np.ndarray] = []
        sample_rate = 16000
        for chunk in audio_ring:
            c_start = float(chunk.get("start_ts", 0.0))
            c_end = float(chunk.get("end_ts", 0.0))
            if c_end <= start_ts or c_start >= end_ts:
                continue
            pcm16 = chunk.get("pcm16")
            if isinstance(pcm16, np.ndarray) and pcm16.size > 0:
                sample_rate = int(chunk.get("sample_rate", sample_rate))
                selected.append(pcm16)
        if not selected:
            return {"player_id": player_id, "audio_files": [], "start_ts": start_ts, "end_ts": end_ts}

        merged = np.concatenate(selected)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        audio_dir = LOG_DIR / f"audio_slices_{session_id}"
        os.makedirs(audio_dir, exist_ok=True)
        out_path = audio_dir / f"speech_slice_{session_id}_{int(start_ts * 1000)}_{ts}.wav"
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(merged.astype(np.int16).tobytes())
        return {"player_id": player_id, "audio_files": [str(out_path)], "start_ts": start_ts, "end_ts": end_ts}

    def process_audio_pcm16(
        self,
        session_id: str,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
        source: str = "backend_audio",
    ) -> Dict:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        audio_path = Path(LOG_DIR / f"{source}_{session_id}_{ts}.wav")
        text = ""
        try:
            with wave.open(str(audio_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())

            if not self.asr_model:
                log_event(
                    self.logger,
                    "backend_audio_asr_skipped",
                    session_id,
                    payload={"reason": "asr_model_unavailable", "source": source, "audio_path": str(audio_path)},
                )
            if self.asr_model:
                try:
                    result = self.asr_model.generate(input=str(audio_path))
                    if isinstance(result, list) and result:
                        text = str(result[0].get("text", ""))
                    elif isinstance(result, dict):
                        text = str(result.get("text", ""))
                except Exception as asr_exc:
                    log_error(self.logger, "asr_exception", session_id, str(asr_exc), payload={"source": source})

            log_event(
                self.logger,
                "asr_result",
                session_id,
                payload={
                    "audio_path": str(audio_path),
                    "asr_text": text,
                    "mime_type": "audio/wav",
                    "source": source,
                    "sample_rate": sample_rate,
                },
            )
        except Exception as exc:
            log_error(self.logger, "audio_decode_exception", session_id, str(exc), payload={"source": source})

        return {"timestamp": datetime.utcnow().isoformat(), "text": text, "source": source}


if __name__ == "__main__":
    asr_model = AutoModel(model="paraformer-zh")
    result = asr_model.generate(input="D:\\Master_Phase\\LLM\\GooseGooseDuck-Agent\\logs\\loopback_32a5b820-5a41-43f3-b913-a6f54637bd1b_20260331_014907_713420.wav")
    print(result)