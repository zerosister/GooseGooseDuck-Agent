import yaml
from backend.utils.pathtool import get_abs_path
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

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
    min_silence_time: float
    max_silence_time: float
    min_segment_time: float
    max_segment_time: float
    vad_threshold: float
    vad_model: str
    spk_model: str

class ASRConfig(BaseModel):
    asr_dir: str
    asr_model: str
    graph: str
    tokens: str
    encoder: str
    decoder: str
    joiner: str
    provider: str

class ModelConfig(BaseModel):
    whisper_model_path: str
    ocr_model_dir: str
    compute_type: str

class VisionConfig(BaseModel):
    mode: str
    target: str
    fps_limit: int

class GameSettingConfig(BaseModel):
    seat_num: int = Field(13, ge=1, le=15)

class AppConfig(BaseModel):
    server: ServerConfig
    audio: AudioConfig
    asr: ASRConfig
    models: ModelConfig
    vision: VisionConfig
    game_setting: GameSettingConfig = Field(default_factory=GameSettingConfig)

def load_config(config_path: str = "config.yaml") -> AppConfig:
    config_path = get_config_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

def get_config_path(config_path: str = "config.yaml"):
    return get_abs_path(config_path)

def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

def get_public_config() -> Dict[str, Any]:
    current = load_config()
    return {
        "server": {
            "host": current.server.host,
            "port": current.server.port,
        },
        "vision": {
            "mode": current.vision.mode,
            "target": current.vision.target,
            "fps_limit": current.vision.fps_limit,
        },
        "game_setting": {
            "seat_num": current.game_setting.seat_num,
        },
    }

def save_public_config(payload: Dict[str, Any], config_path: str = "config.yaml") -> Dict[str, Any]:
    path = get_config_path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw.setdefault("server", {})
    raw.setdefault("vision", {})
    raw.setdefault("game_setting", {})

    if "server" in payload:
        server = payload["server"] or {}
        if "host" in server:
            raw["server"]["host"] = str(server["host"])
        if "port" in server:
            raw["server"]["port"] = int(server["port"])

    if "vision" in payload:
        vision = payload["vision"] or {}
        if "mode" in vision:
            raw["vision"]["mode"] = str(vision["mode"])
        if "target" in vision:
            raw["vision"]["target"] = str(vision["target"])
        if "fps_limit" in vision:
            raw["vision"]["fps_limit"] = int(vision["fps_limit"])

    if "game_setting" in payload:
        game_setting = payload["game_setting"] or {}
        if "seat_num" in game_setting:
            raw["game_setting"]["seat_num"] = int(game_setting["seat_num"])

    validated = AppConfig(**raw)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_model_to_dict(validated), f, allow_unicode=True, sort_keys=False)

    return get_public_config()

# 全局单例
config = load_config()
