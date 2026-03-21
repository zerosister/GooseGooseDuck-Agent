"""
决策 Agent 使用的结构化输入/输出。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


EliminationCause = Literal[
    "vote",
    "kill",
    "explosion",
    "pelican",
    "professional_kill",
    "other",
]


class EliminationRecord(BaseModel):
    victim_id: str = Field(..., description="出局玩家 ID")
    cause: EliminationCause | str = Field(...)
    meeting_index: Optional[int] = None
    round_index: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeetingRoundState(BaseModel):
    round_id: str = Field(...)
    eliminations: list[EliminationRecord] = Field(default_factory=list)
    notes: Optional[str] = None


PlayerAlignment = Literal["goose", "duck", "neutral", "unknown"]


class PlayerRosterEntry(BaseModel):
    player_id: str = Field(...)
    seat_number: int = Field(...)
    color: str = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionContext(BaseModel):
    session_id: str = Field(...)
    self_player_number: int = Field(..., description="当前 assisted 玩家座位号")
    self_player_id: str = Field(..., description="当前 assisted 玩家 ID")
    role_name: str = Field(...)
    alignment: PlayerAlignment | str = Field(default="unknown")
    rounds: list[MeetingRoundState] = Field(default_factory=list)
    player_roster: list[PlayerRosterEntry] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    prior_speech_analysis: str = Field(...)
    speech_suggestion: str = Field(...)
    rag_queries_used: list[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
