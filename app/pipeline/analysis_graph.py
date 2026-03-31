from typing import Any, Dict, List, TypedDict

from app.llm.tongyi_client import TongyiClient
from utils.logger import log_event

from langgraph.graph import END, StateGraph


class AnalysisState(TypedDict):
    session_id: str
    speeches: List[Dict[str, Any]]
    prompt: str
    result: Dict[str, Any]


class AnalysisEngine:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.client = TongyiClient(logger=logger)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AnalysisState)
        workflow.add_node("build_prompt", self._build_prompt_node)
        workflow.add_node("call_tongyi", self._call_tongyi_node)
        workflow.set_entry_point("build_prompt")
        workflow.add_edge("build_prompt", "call_tongyi")
        workflow.add_edge("call_tongyi", END)
        return workflow.compile()

    def _build_prompt_node(self, state: AnalysisState) -> AnalysisState:
        speeches = state.get("speeches", [])
        state["prompt"] = f"speech_count={len(speeches)}"
        log_event(
            self.logger,
            "graph_build_prompt",
            state["session_id"],
            payload={"speech_count": len(speeches)},
        )
        return state

    async def _call_tongyi_node(self, state: AnalysisState) -> AnalysisState:
        result = await self.client.infer_identities(state["session_id"], state.get("speeches", []))
        state["result"] = result
        log_event(
            self.logger,
            "graph_call_tongyi_done",
            state["session_id"],
            payload={"has_result": bool(result)},
        )
        return state

    async def run(self, session_id: str, speeches: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = await self.graph.ainvoke(
            {"session_id": session_id, "speeches": speeches, "prompt": "", "result": {}}
        )
        return out.get("result", {})
        