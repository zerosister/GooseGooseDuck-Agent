from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.model.factory import agent_chat_model
from backend.schemas.contract import IngestionOutput, MemorySummary, PlayerState
from backend.services import rag_query
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import load_summarize_prompt


def _last_ai_text(messages: list[BaseMessage]) -> str:
    """与 decision_agent 一致：取最后一条无 tool_calls 的 AIMessage 文本。"""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        if m.tool_calls:
            continue
        if isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""


def _format_short_memory(ingestions: list[IngestionOutput], recent_n: int) -> str:
    if not ingestions:
        return "(无)"
    window = ingestions[-recent_n:] if recent_n > 0 else ingestions
    lines: list[str] = []
    for io in window:
        sid = io.metadata.get("speaker_id", "?")
        seq = io.sequence_id
        seq_s = f"seq={seq}" if seq is not None else "seq=?"
        content = (io.content or "").strip()
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"[{seq_s}] 说话者={sid}: {content}")
    return "\n".join(lines)


def _build_user_content(user_template: str, prior: str, short: str, ingestion: str) -> str:
    """避免 str.format 与 JSON 花括号冲突，使用 replace。"""
    return (
        user_template.replace("{prior_memory_summary}", prior)
        .replace("{short_memory}", short)
        .replace("{ingestion}", ingestion)
    )


class MemoryAgent:
    def __init__(
        self,
        model=None,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ):
        loaded_sys, loaded_user = load_summarize_prompt()
        self._system_prompt = system_prompt if system_prompt is not None else loaded_sys
        self._user_template = user_template if user_template is not None else loaded_user
        self.model = model if model is not None else agent_chat_model
        self.agent = create_agent(
            model=self.model,
            tools=[rag_query],
            system_prompt=self._system_prompt,
            middleware=[],
        )

    def _memory_conf(self) -> dict:
        return agent_conf.get("memory_agent") or {}

    def _recursion_limit(self) -> int:
        mc = self._memory_conf()
        if mc.get("recursion_limit") is not None:
            return int(mc["recursion_limit"])
        dc = agent_conf.get("decision_agent") or {}
        return int(dc.get("recursion_limit", 25))

    async def process(
        self,
        ingestions: list[IngestionOutput],
        summary: MemorySummary,
        recent_n: int | None = None,
    ) -> MemorySummary:
        if not ingestions:
            return summary

        if recent_n is None:
            recent_n = int(self._memory_conf().get("recent_n", 13))

        last = ingestions[-1]
        prior_json = json.dumps(
            summary.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        short_mem = _format_short_memory(ingestions, recent_n)
        ingestion_text = last.model_dump_json(indent=2, ensure_ascii=False)

        user_content = _build_user_content(
            self._user_template, prior_json, short_mem, ingestion_text
        )
        invoke_config = {"recursion_limit": self._recursion_limit()}
        input_state = {"messages": [HumanMessage(content=user_content)]}

        try:
            result = await self.agent.ainvoke(input_state, invoke_config)
        except (AttributeError, NotImplementedError, TypeError):
            result = await asyncio.to_thread(self.agent.invoke, input_state, invoke_config)

        messages: list[BaseMessage] = list(result.get("messages") or [])
        summary_text = _last_ai_text(messages)
        if not summary_text:
            logger.warning("MemoryAgent: empty model output, falling back to content snippet")
            summary_text = (last.content or "").strip() or "（无文本）"

        summary.timestamp = datetime.now(timezone.utc).isoformat()
        player_state = PlayerState(
            speaker_id=str(last.metadata.get("speaker_id") or "unknown"),
            latest_stance=summary_text,
            emotion_trend="稳定",
        )
        summary.player_summaries.append(player_state)
        return summary
