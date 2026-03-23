from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from typing import Optional

try:
    from funasr import AutoModel  # type: ignore
    from funasr.utils.postprocess_utils import rich_transcription_postprocess  # type: ignore
except Exception:  # pragma: no cover
    AutoModel = None
    rich_transcription_postprocess = None


@dataclass
class ASRConfig:
    model: str = "iic/SenseVoiceSmall"
    vad_model: str = "fsmn-vad"
    sample_rate: int = 44100
    channels: int = 2


class FunASRService:
    def __init__(self, config: Optional[ASRConfig] = None, preloaded_model=None):
        self.config = config or ASRConfig()
        if preloaded_model is not None:
            self._model = preloaded_model
        else:
            if AutoModel is None:
                raise RuntimeError(
                    "FunASR 未安装或不可用。请在 A 工程环境中执行 `pip install -r requirements.txt`，"
                    "或自行安装 `funasr` 相关依赖后再使用 ASR。"
                )
            import torch
            _device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = AutoModel(
                model=self.config.model,
                vad_model=self.config.vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
                device=_device,
            )

    def transcribe_pcm_frames(self, audio_frames: list[bytes]) -> str:
        if not audio_frames:
            return ""

        audio_data = b"".join(audio_frames)
        temp_dir = os.path.join(os.getcwd(), "tmp_asr")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"audio_{int(time.time() * 1000)}.wav")

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio_data)

        try:
            result = self._model.generate(
                input=temp_file,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                if rich_transcription_postprocess is not None:
                    return rich_transcription_postprocess(text)
                return text
            return ""
        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass

