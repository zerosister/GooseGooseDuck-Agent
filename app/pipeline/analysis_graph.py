import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

from app.llm.tongyi_client import TongyiClient
from utils.logger import log_event

from langgraph.graph import END, StateGraph


class AnalysisState(TypedDict):
    session_id: str
    speeches: List[Dict[str, Any]]
    prompt: str
    result: Dict[str, Any]
    seat_map: Dict[int, str]
    current_speaker_num: Optional[int]
    last_speaker_num: Optional[int]
    speech_start_ts: Optional[float]
    pending_speech_event: Optional[Dict[str, Any]]
    frame_path: str
    candidate_speaker_num: Optional[int]
    candidate_speaker_count: int
    speaker_scores: Dict[int, float]
    seat_map_init_fn: Callable[[str, str], Dict[int, str]]
    speaker_detect_fn: Callable[[str, str], Dict[str, Any]]


class AnalysisEngine:
    def __init__(self, logger_1, logger_2) -> None:
        self.logger_1 = logger_1
        self.logger_2 = logger_2
        self.client = TongyiClient(logger=logger_1)
        self.graph = self._build_graph()
        self.realtime_graph = self._build_realtime_graph()
        self._runtime_states: Dict[str, AnalysisState] = {}

    def _build_graph(self):
        """
        LLM 工作的流程图（异步工作）
        """
        workflow = StateGraph(AnalysisState)
        workflow.add_node("build_prompt", self._build_prompt_node)
        workflow.add_node("call_tongyi", self._call_tongyi_node)
        workflow.set_entry_point("build_prompt")
        workflow.add_edge("build_prompt", "call_tongyi")
        workflow.add_edge("call_tongyi", END)
        return workflow.compile()

    def _build_realtime_graph(self):
        """
        实时分析的流程图（同步工作）
        """
        workflow = StateGraph(AnalysisState)
        workflow.add_node("initialize_seat_map", self._initialize_seat_map_node)
        workflow.add_node("detect_speaker_status", self._detect_speaker_status_node)
        workflow.add_node("state_router", self._state_router_node)
        workflow.set_entry_point("initialize_seat_map")
        workflow.add_edge("initialize_seat_map", "detect_speaker_status")
        workflow.add_edge("detect_speaker_status", "state_router")
        workflow.add_edge("state_router", END)
        return workflow.compile()

    def _init_runtime_state(self, session_id: str) -> AnalysisState:
        """
        初始化分析引擎的 AnalysisState 
        """
        return {
            "session_id": session_id,
            "speeches": [],
            "prompt": "",
            "result": {},
            "seat_map": {},
            "current_speaker_num": None,
            "last_speaker_num": None,
            "speech_start_ts": None,
            "pending_speech_event": None,
            "frame_path": "",
            "candidate_speaker_num": None,
            "candidate_speaker_count": 0,
            "speaker_scores": {},
        }

    def _build_prompt_node(self, state: AnalysisState) -> AnalysisState:
        """
        构建 prompt 的节点
        """
        speeches = state.get("speeches", [])
        state["prompt"] = f"speech_count={len(speeches)}"
        log_event(
            self.logger_1,
            "graph_build_prompt",
            state["session_id"],
            payload={"speech_count": len(speeches)},
        )
        return state

    async def _call_tongyi_node(self, state: AnalysisState) -> AnalysisState:
        """
        异步调用 LLM 的节点
        """
        result = await self.client.infer_identities(state["session_id"], state.get("speeches", []))
        state["result"] = result
        log_event(
            self.logger_1,
            "graph_call_tongyi_done",
            state["session_id"],
            payload={"has_result": bool(result)},
        )
        return state

    def _initialize_seat_map_node(self, state: AnalysisState) -> AnalysisState:
        """
        初始化座位图的节点
        """
        init_fn = state.get("seat_map_init_fn")
        log_event(self.logger_2, "init_seat_map_start", state["session_id"], payload={"init_fn": init_fn.__name__ if init_fn else None})
        if state.get("seat_map"):
            log_event(self.logger_2, "seat_map_already_initialized", state["session_id"], payload={"seat_map": state["seat_map"]})
            return state
        if callable(init_fn):
            log_event(self.logger_2, "init_seat_map_callable", state["session_id"], payload={"init_fn": init_fn.__name__})
            seat_map = init_fn(state["session_id"], state.get("frame_path", ""))
            state["seat_map"] = seat_map or {}
            log_event(
                self.logger_2,
                "seat_map_initialized",
                state["session_id"],
                payload={"seat_count": len(state["seat_map"])},
            )
        else:
            log_event(self.logger_2, "init_seat_map_not_callable", state["session_id"], payload={"init_fn": init_fn})
        log_event(self.logger_2, "init_seat_map_end", state["session_id"], payload={"seat_map": state["seat_map"]})
        return state

    def _detect_speaker_status_node(self, state: AnalysisState) -> AnalysisState:
        """
        检测当前帧谁在说话
        """
        detect_fn = state.get("speaker_detect_fn")
        speaker_num: Optional[int] = None
        scores: Dict[int, float] = {}
        if callable(detect_fn):
            detect_res = detect_fn(state["session_id"], state.get("frame_path", ""))
            speaker_num = detect_res.get("speaker_num")
            scores = detect_res.get("scores", {})

        if speaker_num == state.get("candidate_speaker_num"):
            state["candidate_speaker_count"] = int(state.get("candidate_speaker_count", 0)) + 1
        else:
            state["candidate_speaker_num"] = speaker_num
            state["candidate_speaker_count"] = 1

        if state.get("candidate_speaker_count", 0) >= 2:
            state["current_speaker_num"] = state.get("candidate_speaker_num")

        state["speaker_scores"] = scores
        log_event(
            self.logger_2,
            "speaker_detected",
            state["session_id"],
            payload={
                "candidate_speaker_num": state.get("candidate_speaker_num"),
                "candidate_speaker_count": state.get("candidate_speaker_count"),
                "current_speaker_num": state.get("current_speaker_num"),
            },
        )
        return state

    def _state_router_node(self, state: AnalysisState) -> AnalysisState:
        """
        处理发言状态的切换逻辑。生成发言事件
        """
        now_ts = time.time()
        prev = state.get("last_speaker_num")
        curr = state.get("current_speaker_num")
        state["pending_speech_event"] = None

        if prev is None and curr is not None:
            state["speech_start_ts"] = now_ts
        elif prev is not None and curr != prev:
            start_ts = state.get("speech_start_ts") or now_ts
            end_ts = now_ts
            speaker_name = state.get("seat_map", {}).get(prev, f"Unknown_{prev:02d}")
            state["pending_speech_event"] = {
                "speaker_num": prev,
                "speaker_name": speaker_name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_ms": int(max(0.0, (end_ts - start_ts)) * 1000),
            }
            state["speech_start_ts"] = now_ts if curr is not None else None
            log_event(
                self.logger_2,
                "speech_event_created",
                state["session_id"],
                payload=state["pending_speech_event"],
            )

        state["last_speaker_num"] = curr
        return state

    async def run(self, session_id: str, speeches: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = await self.graph.ainvoke(
            {"session_id": session_id, "speeches": speeches, "prompt": "", "result": {}}
        )
        return out.get("result", {})

    def process_frame(
        self,
        session_id: str,
        frame_path: str,
        seat_map_init_fn: Callable[[str, str], Dict[int, str]],
        speaker_detect_fn: Callable[[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        实时分析的入口函数
        """
        state = self._runtime_states.setdefault(session_id, self._init_runtime_state(session_id))
        graph_input = dict(state)
        graph_input["frame_path"] = frame_path
        graph_input["seat_map_init_fn"] = seat_map_init_fn
        graph_input["speaker_detect_fn"] = speaker_detect_fn
        out = self.realtime_graph.invoke(graph_input)
        out.pop("seat_map_init_fn", None)
        out.pop("speaker_detect_fn", None)
        self._runtime_states[session_id] = out
        return out

    def reset_session(self, session_id: str) -> None:
        self._runtime_states.pop(session_id, None)