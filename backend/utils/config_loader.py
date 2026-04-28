import yaml
from backend.utils.pathtool import get_abs_path
from pydantic import BaseModel, Field
from typing import List, Optional

class ServerConfig(BaseModel):
    host: str
    port: int
    debug: bool

class AudioConfig(BaseModel):
    sample_rate: int
    channels: int
    device_index: Optional[int] = Field(None)
    save_path: str
    save_temp: bool
    vad_threshold: float
    min_silence_time: float
    max_silence_time: float
    min_segment_time: float
    max_segment_time: float
    vad_threshold: float
    vad_model_path: str

class ModelConfig(BaseModel):
    whisper_model_path: str
    ocr_model_dir: str
    compute_type: str

class ASRConfig(BaseModel):
    model_size: str
    device: str
    compute_type: str

class AppConfig(BaseModel):
    server: ServerConfig
    audio: AudioConfig
    models: ModelConfig
    vision: dict # 也可以继续细化
    asr: ASRConfig

def load_config(config_path: str = "config.yaml") -> AppConfig:
    config_path = get_abs_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

# 全局单例
config = load_config()