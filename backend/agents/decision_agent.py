from __future__ import annotations

import asyncio
import json
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.decision import DecisionContext, DecisionResult
from backend.model.factory import agent_chat_model
from backend.services import rag_query
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import load_decision_prompts


def _parse_marked_output(text: str) -> tuple[str, str, str, list[str]]:
    """解析模型按【前序发言分析】/【玩家身份推测】/【发言建议】三段标记的输出。"""
    warnings: list[str] = []
    m_prior = "【前序发言分析】"
    m_identity = "【玩家身份推测】"
    m_sug = "【发言建议】"

    if not (text or "").strip():
        return "", "", "", warnings

    raw = text.strip()
    i_prior = raw.find(m_prior)
    i_id = raw.find(m_identity)
    i_sug = raw.find(m_sug)

    prior = ""
    identity = ""
    suggestion = ""

    if i_prior >= 0:
        if i_id >= 0 and i_id > i_prior:
            prior = raw[i_prior + len(m_prior) : i_id].strip()
        elif i_sug >= 0 and i_sug > i_prior:
            prior = raw[i_prior + len(m_prior) : i_sug].strip()
        else:
            prior = raw[i_prior + len(m_prior) :].strip()
    else:
        warnings.append("未找到【前序发言分析】标记")

    if i_id >= 0:
        if i_sug >= 0 and i_sug > i_id:
            identity = raw[i_id + len(m_identity) : i_sug].strip()
        else:
            identity = raw[i_id + len(m_identity) :].strip()
    elif i_prior >= 0 and i_sug >= 0:
        warnings.append("未找到【玩家身份推测】标记（兼容旧两段格式时身份推测为空）")

    if i_sug >= 0:
        suggestion = raw[i_sug + len(m_sug) :].strip()
    else:
        warnings.append("未找到【发言建议】标记")

    if i_prior < 0 and i_id < 0 and i_sug < 0:
        warnings.append("输出未包含分段标记，全文暂作发言建议展示")
        suggestion = raw

    return prior, identity, suggestion, warnings


def _collect_rag_queries(messages: list[BaseMessage]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in messages:
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            if tc.get("name") != "rag_query":
                continue
            args = tc.get("args") or {}
            q = args.get("query")
            if isinstance(q, str) and q.strip() and q not in seen:
                seen.add(q)
                out.append(q.strip())
    return out


def _last_ai_text(messages: list[BaseMessage]) -> str:
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


def _build_debug_prompt(
    system_prompt: str, user_content: str, messages: list[BaseMessage]
) -> str:
    lines = [
        "### System\n",
        system_prompt,
        "\n\n### User\n",
        user_content,
        "\n\n### Tool trace\n",
    ]
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"{tc.get('name')}: {tc.get('args')}\n")
    return "".join(lines)


class DecisionAgent:
    """
    决策：LangChain create_agent 工具循环，按需调用 rag_query；
    system / user 模板由 prompts.yaml 配置。
    """

    def __init__(
        self,
        model=None,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ):
        loaded_sys, loaded_user = load_decision_prompts()
        self._system_prompt = (
            system_prompt if system_prompt is not None else loaded_sys
        )
        self._user_template = (
            user_template if user_template is not None else loaded_user
        )
        self.model = model or agent_chat_model
        self.agent = create_agent(
            model=self.model,
            tools=[rag_query],
            system_prompt=self._system_prompt,
            middleware=[],
        )
        logger.info("DecisionAgent initialized (create_agent + rag_query)")

    def _decision_conf(self) -> dict:
        return agent_conf.get("decision_agent", {})

    async def run(
        self,
        ctx: DecisionContext,
        memory: MemorySummary,
        ingestions: list[IngestionOutput] | None = None,
    ) -> DecisionResult:
        conf = self._decision_conf()
        recent_n = int(conf.get("recent_ingestion_n", 20))
        recursion_limit = int(conf.get("recursion_limit", 25))

        warnings: list[str] = []

        recent = list(ingestions or [])
        if recent_n > 0 and len(recent) > recent_n:
            recent = recent[-recent_n:]
        recent_json = json.dumps(
            [x.model_dump() for x in recent],
            ensure_ascii=False,
            indent=2,
        )

        payload = {
            "decision_context": ctx.model_dump_json(indent=2, ensure_ascii=False),
            "memory_summary": memory.model_dump_json(indent=2, ensure_ascii=False),
            "recent_ingestions": recent_json,
        }
        user_content = self._user_template.format(**payload)
        logger.info("DecisionAgent user payload keys: %s", list(payload.keys()))

        invoke_config = {"recursion_limit": recursion_limit}
        input_state = {"messages": [HumanMessage(content=user_content)]}

        try:
            result = await self.agent.ainvoke(input_state, invoke_config)
        except (AttributeError, NotImplementedError, TypeError):
            result = await asyncio.to_thread(
                self.agent.invoke, input_state, invoke_config
            )

        messages: list[BaseMessage] = list(result.get("messages") or [])
        raw = _last_ai_text(messages)
        if not raw:
            warnings.append("模型未返回可解析的文本内容")
        rag_queries_used = _collect_rag_queries(messages)
        prior, identity, suggestion, parse_warnings = _parse_marked_output(raw or "")
        warnings.extend(parse_warnings)

        debug_prompt = _build_debug_prompt(
            self._system_prompt, user_content, messages
        )

        return DecisionResult(
            prior_speech_analysis=prior,
            identity_inference=identity,
            speech_suggestion=suggestion,
            rag_queries_used=rag_queries_used,
            warnings=warnings,
            debug_prompt=debug_prompt,
        )


if __name__ == "__main__":
    sample = """【前序发言分析】
a
【玩家身份推测】
3号-测试-未知
【发言建议】
b
"""
    p, i, s, w = _parse_marked_output(sample)
    print("parse:", p, i, s, w)
