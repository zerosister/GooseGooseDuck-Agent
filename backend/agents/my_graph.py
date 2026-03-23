"""
LangGraph：仅 memory_agent（每人发言结束后跑一轮）。

「决策信号」与 DecisionContext 均独立于 checkpoint：前端在适当时机调用
`run_decision(thread_id, decision_context)`，其中 decision_context 由业务侧早已准备好的数据组装，
与图状态中的 ingestions/summary 组合后调用 DecisionAgent。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from backend.agents.decision_agent import DecisionAgent
from backend.agents.memory_agent import MemoryAgent
from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.decision import DecisionContext, DecisionResult
from backend.schemas.graph_state import MemoryDecisionState


async def memory_node(state: MemoryDecisionState) -> dict:
    """调用 MemoryAgent 更新摘要。"""
    agent = MemoryAgent()
    if not state.ingestions:
        return {}
    summary = await agent.process(
        state.ingestions,
        state.summary
        if state.summary
        else MemorySummary(
            session_id=state.ingestions[0].session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return {"summary": summary}


class MemoryGraph:
    """
    START → memory_agent → END。

    决策：收到独立「决策信号」且已准备好 `DecisionContext` 时调用
    `run_decision(thread_id, decision_context)`（不依赖 checkpoint 存 context）。
    """

    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.graph = self._build()
        self.decision_agent = DecisionAgent()

    def _build(self):
        builder = StateGraph(MemoryDecisionState)
        builder.add_node("memory_agent", memory_node)
        builder.add_edge(START, "memory_agent")
        builder.add_edge("memory_agent", END)
        return builder.compile(checkpointer=self.checkpointer)

    async def ainvoke(self, ingestion: IngestionOutput, thread_id: str) -> Any:
        """每条发言结束后调用，仅更新 ingestions / summary。"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.ainvoke({"ingestions": [ingestion]}, config)

    async def run_decision(
        self,
        thread_id: str,
        decision_context: DecisionContext,
    ) -> Optional[DecisionResult]:
        """
        独立决策信号：从 checkpoint 读取当前 summary / ingestions，
        与调用方传入的 decision_context 一起交给 DecisionAgent，并写回 decision_result。
        """
        config = {"configurable": {"thread_id": thread_id}}
        snap = await self.graph.aget_state(config)
        raw = snap.values
        if not raw:
            return None
        try:
            state = MemoryDecisionState.model_validate(raw)
        except Exception:
            return None
        if not state.summary:
            return None
        result = await self.decision_agent.run(
            decision_context,
            state.summary,
            state.ingestions,
        )
        update_fn = getattr(self.graph, "aupdate_state", None)
        if update_fn is not None:
            await update_fn(
                config,
                {"decision_result": result},
                as_node="memory_agent",
            )
        else:
            self.graph.update_state(
                config,
                {"decision_result": result},
                as_node="memory_agent",
            )
        return result


MemoryDecisionGraph = MemoryGraph
