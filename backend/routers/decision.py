"""Memory + decision API (/api/v1/*)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend import app_state
from backend.schemas.contract import IngestionOutput
from backend.schemas.decision import DecisionContext, DecisionResult
from backend.schemas.graph_state import MemoryDecisionState
from backend.services.situation_sketch import schedule_situation_sketch_after_ingestion
from backend.utils.situation_context import (
    get_situation_sketch,
    get_situation_sketch_narrative,
)

logger = logging.getLogger("ggd-b")

router = APIRouter(tags=["memory"])


async def execute_decision_stream(
    ctx: DecisionContext,
) -> AsyncIterator[dict[str, Any]]:
    """流式：草稿 + 规则判官 + 修订全阶段；最后 yield type=done（含最终 DecisionResult）。"""
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")

    speaker_filter = ctx.extra.get("speaker_filter") if ctx.extra else None

    try:
        config = {"configurable": {"thread_id": ctx.session_id}}
        snap = await g.graph.aget_state(config)
        raw = snap.values
        if not raw:
            raise HTTPException(
                status_code=400, detail="No memory data for this session."
            )

        state = MemoryDecisionState.model_validate(raw)
        if not state.summary:
            raise HTTPException(
                status_code=400, detail="No summary yet. Feed ingestions first."
            )

        ingestions = state.ingestions
        if speaker_filter and isinstance(speaker_filter, list):
            ingestions = [
                i
                for i in ingestions
                if i.metadata.get("speaker_id") in speaker_filter
            ]
            if not ingestions:
                raise HTTPException(
                    status_code=400,
                    detail="No ingestions match the selected speakers.",
                )

        draft_final: DecisionResult | None = None
        async for ev in g.decision_agent.run_draft_stream(
            ctx,
            state.summary,
            ingestions,
            situation_sketch=get_situation_sketch(state),
            situation_sketch_narrative=get_situation_sketch_narrative(state),
        ):
            if ev.get("type") == "draft_complete":
                raw_dr = ev.get("result")
                draft_final = (
                    DecisionResult.model_validate(raw_dr)
                    if isinstance(raw_dr, dict)
                    else raw_dr
                )
            out = dict(ev)
            if "phase" not in out:
                out["phase"] = "draft"
            yield out

        if draft_final is None:
            raise HTTPException(
                status_code=500, detail="Draft stream produced no result."
            )

        final = draft_final
        async for ev in g.run_decision_critic_loop_stream(
            ctx,
            state.summary,
            ingestions,
            draft_final,
            situation_sketch=get_situation_sketch(state),
            situation_sketch_narrative=get_situation_sketch_narrative(state),
        ):
            yield ev
            if ev.get("type") == "decision_critic_done" and isinstance(
                ev.get("result"), dict
            ):
                final = DecisionResult.model_validate(ev["result"])

        await g.persist_decision_result(ctx.session_id, final)
        yield {"type": "done", "result": final.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Decision stream failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/v1/status")
async def get_status_v1(
    session_id: str | None = Query(
        None,
        description="若提供，则附带该会话的轻量局势笔记等 checkpoint 字段",
    ),
):
    base: dict = {"status": "running" if app_state.graph else "not_initialized"}
    if session_id and app_state.graph:
        config = {"configurable": {"thread_id": session_id}}
        try:
            snap = await app_state.graph.graph.aget_state(config)
            raw = snap.values
            if raw:
                st = MemoryDecisionState.model_validate(raw)
                base["situation_sketch"] = st.situation_sketch
                base["situation_sketch_narrative"] = st.situation_sketch_narrative
                base["sketch_updated_at"] = st.sketch_updated_at
                base["sketch_warnings"] = st.sketch_warnings
        except Exception:
            logger.debug("status: no state for session_id=%s", session_id, exc_info=True)
    return base


@router.post("/api/v1/ingestion")
async def receive_ingestion(ingestion: IngestionOutput):
    """Receive an IngestionOutput and feed it into MemoryGraph."""
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph not initialized")
    try:
        await g.ainvoke(ingestion, ingestion.session_id)
        await schedule_situation_sketch_after_ingestion(g, ingestion.session_id)
    except Exception as e:
        logger.error("Failed to process ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("Ingestion processed: session=%s seq=%s", ingestion.session_id, ingestion.sequence_id)
    return {"status": "ok", "session_id": ingestion.session_id}


