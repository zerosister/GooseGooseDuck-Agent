"""
B-side FastAPI server: Memory + Decision Agent HTTP API.

Receives IngestionOutput from A-side, accumulates memory via MemoryGraph,
and provides a decision endpoint for generating speech analysis and suggestions.

Run from work_B directory:
    set PYTHONPATH=.
    python -m backend.main
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

_WORK_B = Path(__file__).resolve().parent.parent
if str(_WORK_B) not in sys.path:
    sys.path.insert(0, str(_WORK_B))

from backend.agents.my_graph import MemoryGraph
from backend.schemas.contract import IngestionOutput
from backend.schemas.decision import DecisionContext, DecisionResult
from backend.services.meeting_memory_service import ALLOWED_MSGPACK_MODULES, ShortTermMemoryStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from utils.path_tool import get_abs_path

logger = logging.getLogger("ggd-b")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_graph: MemoryGraph | None = None
_conn: aiosqlite.Connection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _conn

    store = ShortTermMemoryStore()
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    _conn = await aiosqlite.connect(get_abs_path(store.DB_URI))
    saver = AsyncSqliteSaver(_conn, serde=serde)
    _graph = MemoryGraph(saver)
    logger.info("MemoryGraph initialized (SQLite checkpoint ready)")
    yield

    if _conn:
        await _conn.close()
    logger.info("Connection closed")


app = FastAPI(title="GGD-B Memory+Decision API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/status")
async def get_status():
    return {"status": "running" if _graph else "not_initialized"}


@app.post("/api/v1/ingestion")
async def receive_ingestion(ingestion: IngestionOutput):
    """Receive an IngestionOutput from A-side and feed it into MemoryGraph."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")
    try:
        await _graph.ainvoke(ingestion, ingestion.session_id)
    except Exception as e:
        logger.error("Failed to process ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("Ingestion processed: session=%s seq=%s", ingestion.session_id, ingestion.sequence_id)
    return {"status": "ok", "session_id": ingestion.session_id}


@app.post("/api/v1/decision", response_model=DecisionResult)
async def run_decision(ctx: DecisionContext):
    """Trigger decision: read accumulated memory, run RAG + LLM, return analysis and suggestion."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")

    speaker_filter = ctx.extra.get("speaker_filter") if ctx.extra else None

    try:
        config = {"configurable": {"thread_id": ctx.session_id}}
        snap = await _graph.graph.aget_state(config)
        raw = snap.values
        if not raw:
            raise HTTPException(status_code=400, detail="No memory data for this session.")

        from backend.schemas.graph_state import MemoryDecisionState
        state = MemoryDecisionState.model_validate(raw)
        if not state.summary:
            raise HTTPException(status_code=400, detail="No summary yet. Feed ingestions first.")

        ingestions = state.ingestions
        if speaker_filter and isinstance(speaker_filter, list):
            ingestions = [i for i in ingestions if i.metadata.get("speaker_id") in speaker_filter]
            if not ingestions:
                raise HTTPException(status_code=400, detail="No ingestions match the selected speakers.")

        result = await _graph.decision_agent.run(ctx, state.summary, ingestions)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Decision failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return result


if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="127.0.0.1", port=9889, log_level="info")
