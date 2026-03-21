# pydantic 是用于数据验证的库，可在运行时进行类型检查
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional
from datetime import datetime
import operator

class IngestionOutput(BaseModel):
    """读入 Agent 输出，B 的输入。A 负责填充，B 负责消费。"""
    type: Literal["speech", "image"]
    # Field 为精细调控 "..."告诉Pydantic这个字段为强制项
    # description 为字段的描述，在生成项目文档时则会自动添加到文档中
    content: str = Field(..., description="转写文本 或 图像描述")
    # default_factory 可以调用函数，生成动态默认值
    metadata: dict = Field(default_factory=dict, description="扩展信息")
    timestamp: str = Field(..., description="ISO 8601 时间戳")
    session_id: str = Field(..., description="会议/会话 ID，用于关联同一场会议")
    sequence_id: Optional[int] = Field(None, description="同 session 内顺序号，便于 B 排序")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "speech",
                "content": "我怀疑3号是鸭子，他在锅炉房附近鬼鬼祟祟",
                "metadata": {
                    "speaker_id": "4",
                    "speaker_confidence": 0.92,
                    "emotion_summary": "语气坚定、略带怀疑",
                    "sentence_index": 1,
                    "is_final": True
                },
                "timestamp": "2025-03-18T10:30:00.000Z",
                "session_id": "meeting_abc123",
                "sequence_id": 42
            }
        }
        
class PlayerState(BaseModel):
    speaker_id: str
    latest_stance: str = Field(..., description="该玩家最新的立场/观点")
    key_claims: list[str] = Field(default_factory=list, description="该玩家发表过的关键指控")
    emotion_trend: str = Field(..., description="情绪走向（如：从冷静变为激进）")

class MemorySummary(BaseModel):
    session_id: str
    # 你的核心想法：按人汇总的列表
    player_summaries: list[PlayerState] = Field(default_factory=list)

    # 辅助决策的信息
    recent_events: list[str] = Field(default_factory=list, description="最近发生的大事（如：某人被投出）")
    # pydantic 中，字符串不接受 None，所以需要使用 Optional[str]
    overall_atmosphere: Optional[str] = Field(None, description="当前的整体讨论氛围")
    timestamp: str
    
class MemoryTestState(BaseModel):
    # 使用 Annotated 和 operator.add 标记这是一个可累积的列表
    # 每次 ainvoke 传入 [new_ingestion] 时，它会自动追加到原有列表末尾
    ingestions: Annotated[list[IngestionOutput], operator.add] = Field(
        default_factory=list, 
        description="本局所有历史发言记录（短期记忆）"
    )
    
    summary: Optional[MemorySummary] = Field(
        None,
        description="基于当前所有 ingestions 生成的最新摘要，为空则表示第一次调用"
    )