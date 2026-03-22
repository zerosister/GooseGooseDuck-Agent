from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class IngestionOutput(BaseModel):
    """读入 Agent 输出，B 的输入。A 负责填充，B 负责消费。"""

    type: Literal["speech", "image"]
    content: str = Field(..., description="转写文本 或 图像描述")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展信息")
    timestamp: str = Field(..., description="ISO 8601 时间戳")
    session_id: str = Field(..., description="会议/会话 ID，用于关联同一场会议")
    sequence_id: Optional[int] = Field(None, description="同 session 内顺序号，便于 B 排序")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "speech",
                "content": "我怀疑3号是鸭子，他在锅炉房附近鬼鬼祟祟",
                "metadata": {
                    "speaker_id": "3",
                    "speaker_confidence": 0.92,
                    "emotion_summary": "语气坚定、略带怀疑",
                    "sentence_index": 1,
                    "is_final": True,
                },
                "timestamp": "2025-03-18T10:30:00.000Z",
                "session_id": "meeting_abc123",
                "sequence_id": 42,
            }
        }
    }


class DecisionOutput(BaseModel):
    """决策 Agent 输出，推送给前端。B 负责填充。"""

    session_id: str
    suggestion_type: Literal["speak", "vote", "respond", "general"]
    content: str = Field(..., description="建议文本")
    structured: Optional[dict[str, Any]] = Field(None, description="结构化建议，如投票对象、发言要点")
    timestamp: str
    trigger: Optional[str] = Field(None, description="触发来源：speech/image/summary")

