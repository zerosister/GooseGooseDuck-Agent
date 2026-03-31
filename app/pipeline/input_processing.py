import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional
import wave

import numpy as np

from PIL import Image

from utils.logger import LOG_DIR, log_error, log_event

from rapidocr_onnxruntime import RapidOCR
from funasr import AutoModel


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
            return {"timestamp": datetime.utcnow().isoformat(), "text": text, "source": "ocr"}
        except Exception as exc:
            log_error(self.logger, "ocr_exception", session_id, str(exc))
            return {"timestamp": datetime.utcnow().isoformat(), "text": "", "source": "ocr"}

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