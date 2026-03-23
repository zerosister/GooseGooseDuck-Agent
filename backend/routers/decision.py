"""Memory + decision API (/api/v1/*)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend import app_state
from backend.schemas.contract import IngestionOutput
from backend.schemas.decision import DecisionContext, DecisionResult
from backend.schemas.graph_state import MemoryDecisionState

logger = logging.getLogger("ggd-b")

router = APIRouter(tags=["memory"])


async def execute_decision(ctx: DecisionContext) -> DecisionResult:
    """Run decision agent; shared by HTTP and ingestion proxy."""
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")

    speaker_filter = ctx.extra.get("speaker_filter") if ctx.extra else None

    try:
        config = {"configurable": {"thread_id": ctx.session_id}}
        snap = await g.graph.aget_state(config)
        raw = snap.values
        if not raw:
            raise HTTPException(status_code=400, detail="No memory data for this session.")

        state = MemoryDecisionState.model_validate(raw)
        if not state.summary:
            raise HTTPException(status_code=400, detail="No summary yet. Feed ingestions first.")

        ingestions = state.ingestions
        if speaker_filter and isinstance(speaker_filter, list):
            ingestions = [i for i in ingestions if i.metadata.get("speaker_id") in speaker_filter]
            if not ingestions:
                raise HTTPException(status_code=400, detail="No ingestions match the selected speakers.")

        result = await g.decision_agent.run(ctx, state.summary, ingestions)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Decision failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


@router.get("/api/v1/status")
async def get_status_v1():
    return {"status": "running" if app_state.graph else "not_initialized"}


@router.post("/api/v1/ingestion")
async def receive_ingestion(ingestion: IngestionOutput):
    """Receive an IngestionOutput and feed it into MemoryGraph."""
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")
    try:
        await g.ainvoke(ingestion, ingestion.session_id)
    except Exception as e:
        logger.error("Failed to process ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("Ingestion processed: session=%s seq=%s", ingestion.session_id, ingestion.sequence_id)
    return {"status": "ok", "session_id": ingestion.session_id}


@router.post("/api/v1/decision", response_model=DecisionResult)
async def run_decision(ctx: DecisionContext):
    """Trigger decision: read accumulated memory, run RAG + LLM."""
    return await execute_decision(ctx)
