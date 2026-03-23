"""LangGraph checkpoint：只累积发言与摘要；决策上下文由 `run_decision(..., decision_context)` 外部传入。"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.decision import DecisionResult


class MemoryDecisionState(BaseModel):
    ingestions: Annotated[list[IngestionOutput], operator.add] = Field(
        default_factory=list,
        description="本局发言记录（累积）",
    )
    summary: Optional[MemorySummary] = Field(
        None,
        description="MemoryAgent 维护的摘要",
    )
    decision_result: Optional[DecisionResult] = Field(
        None,
        description="最近一次 decision_agent 输出（仅缓存结果，与触发信号无关）",
    )
