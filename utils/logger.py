import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_LOG_DIR = LOG_DIR / "sessions"
SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
DAY_LOG_DIR = LOG_DIR / "daily"
DAY_LOG_DIR.mkdir(parents=True, exist_ok=True)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_type"):
            payload["event_type"] = record.event_type
        if hasattr(record, "session_id"):
            payload["session_id"] = record.session_id
        if hasattr(record, "payload"):
            payload["payload"] = record.payload
        if hasattr(record, "error"):
            payload["error"] = record.error
        return json.dumps(payload, ensure_ascii=False)


def _build_handler(filename: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    return handler


def get_logger(name: str = "ggd") -> logging.Logger:
    logger = logging.getLogger(name)
    
    # 1. 获取顶级名称（例如 "ggd.engine" -> "ggd"）
    top_level_name = name.split('.')[0]
    top_logger = logging.getLogger(top_level_name)
    
    # 2. 核心逻辑：如果顶级 Logger 还没有配置 Handler，就给它配置
    # 这样无论你请求的是 "ggd" 还是 "ggd.engine"，都会确保 "ggd" 已经被初始化
    if not top_logger.handlers:
        top_logger.setLevel(logging.INFO)
        # 假设 _build_handler 是你定义的返回 Handler 的函数
        handler = _build_handler(f"{top_level_name}.log") 
        top_logger.addHandler(handler)
        top_logger.propagate = False 
        
    return logger


def log_event(
    logger: logging.Logger,
    event_type: str,
    session_id: str,
    payload: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    # 先判断当前 logger 是否开启了此级别
    if not logger.isEnabledFor(level):
        return
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event_type": event_type,
        "session_id": session_id,
        "payload": payload or {},
    }
    logger.log(level, f"{event_type}", extra=record)
    _write_json_line(SESSION_LOG_DIR / f"{session_id}.jsonl", {"level": "INFO", **record})
    _write_json_line(DAY_LOG_DIR / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl", {"level": "INFO", **record})


def log_error(
    logger: logging.Logger,
    event_type: str,
    session_id: str,
    error: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event_type": event_type,
        "session_id": session_id,
        "payload": payload or {},
        "error": error,
    }
    logger.error(f"{event_type}: {error}", extra=record)
    _write_json_line(SESSION_LOG_DIR / f"{session_id}.jsonl", {"level": "ERROR", **record})
    _write_json_line(DAY_LOG_DIR / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl", {"level": "ERROR", **record})


def _write_json_line(path: Path, data: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
