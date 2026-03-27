from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.agents.middleware import (
    DecisionAgentContext,
    before_agent_middleware,
    decision_phase_prompt_middleware,
    monitor_tool,
    situation_sketch_model_context,
)
from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.graph_state import SituationSketch
from backend.schemas.decision import DecisionResult, RuleCriticReview
from backend.model.factory import get_decision_chat_model
from backend.services import rag_query
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import load_decision_prompts, load_decision_revise_prompts
from backend.agents import last_ai_text, stream_deltas_from_chunk


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


def _require_stream_model(model: BaseChatModel) -> None:
    if not callable(getattr(model, "astream", None)):
        raise RuntimeError(
            "DecisionAgent 需要支持流式 (astream) 的 Chat 模型；当前模型不可用。"
        )


class DecisionAgent:
    """
    决策：仅流式 create_agent + rag_query；草案/修订通过 runtime context 切换 system prompt。
    """

    def __init__(
        self,
        model=None,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ):
        loaded_sys, loaded_user = load_decision_prompts()
        rev_sys, rev_user = load_decision_revise_prompts()
        self._system_prompt = (
            system_prompt if system_prompt is not None else loaded_sys
        )
        self._user_template = (
            user_template if user_template is not None else loaded_user
        )
        self._revise_system = rev_sys
        self._revise_user_template = rev_user
        stream_model = model if model is not None else get_decision_chat_model()
        if not isinstance(stream_model, BaseChatModel):
            raise TypeError("DecisionAgent model must be a BaseChatModel instance")
        _require_stream_model(stream_model)
        self.model = stream_model
        self.agent = create_agent(
            model=self.model,
            tools=[rag_query],
            system_prompt=self._system_prompt,
            context_schema=DecisionAgentContext,
            middleware=[
                situation_sketch_model_context,
                decision_phase_prompt_middleware(
                    self._system_prompt, self._revise_system
                ),
                monitor_tool,
                before_agent_middleware,
            ],
        )
        logger.info("DecisionAgent initialized (stream-only create_agent + rag_query)")

    def _decision_conf(self) -> dict:
        return agent_conf.get("decision_agent", {})

    def _draft_user_payload(
        self,
        memory: MemorySummary,
    ) -> tuple[str, str]:
        payload = {
            "memory_summary": memory.model_dump_json(indent=2, ensure_ascii=False),
        }
        user_content = self._user_template.format(**payload)
        return user_content

    async def run_draft_stream(
        self,
        memory: MemorySummary,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """首轮决策草稿流式：yield 增量事件，最后 yield draft_complete 含 DecisionResult。"""
        conf = self._decision_conf()
        recursion_limit = int(conf.get("recursion_limit", 25))
        user_content = self._draft_user_payload(memory)
        logger.info("DecisionAgent user payload keys: memory_summary")

        invoke_config = {"recursion_limit": recursion_limit}
        input_state = {"messages": [HumanMessage(content=user_content)]}
        phase_ctx = DecisionAgentContext(
            phase="draft",
            situation_sketch=situation_sketch,
            situation_sketch_narrative=situation_sketch_narrative,
        )

        messages: list[BaseMessage] | None = None
        try:
            async for event in self.agent.astream_events(
                input_state,
                invoke_config,
                version="v2",
                context=phase_ctx,
            ):
                et = event.get("event")
                if et == "on_chat_model_stream":
                    chunk = (event.get("data") or {}).get("chunk")
                    for kind, delta in stream_deltas_from_chunk(chunk):
                        yield {"type": kind, "delta": delta}
                elif et == "on_tool_start":
                    name = (event.get("data") or {}).get("name") or event.get("name")
                    yield {"type": "tool_start", "name": str(name or "")}
                elif et == "on_tool_end":
                    name = (event.get("data") or {}).get("name") or event.get("name")
                    yield {"type": "tool_end", "name": str(name or "")}
                elif et == "on_chain_end":
                    out = (event.get("data") or {}).get("output")
                    if isinstance(out, dict) and out.get("messages") is not None:
                        messages = list[BaseMessage](out["messages"])
        except Exception as e:
            logger.error("DecisionAgent.run_draft_stream: astream_events failed: %s", e)
            raise RuntimeError(
                f"决策草稿流式失败（不支持非流式回退）。原始错误：{e}"
            ) from e

        if messages is None:
            raise RuntimeError(
                "决策草稿流未返回最终 messages，请检查 LangGraph 与模型流式配置。"
            )

        raw = last_ai_text(messages)
        warnings: list[str] = []
        if not raw:
            warnings.append("模型未返回可解析的文本内容")
        prior, identity, suggestion, parse_warnings = _parse_marked_output(raw or "")
        warnings.extend(parse_warnings)
        debug_prompt = _build_debug_prompt(
            self._system_prompt, user_content, messages
        )
        dr = DecisionResult(
            prior_speech_analysis=prior,
            identity_inference=identity,
            speech_suggestion=suggestion,
            warnings=warnings,
            debug_prompt=debug_prompt,
        )
        yield {"type": "draft_complete", "result": dr.model_dump()}

    async def revise_from_critic_stream(
        self,
        result: DecisionResult,
        review: RuleCriticReview,
        iteration: int = 0,
        situation_sketch: SituationSketch | None = None,
        situation_sketch_narrative: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """根据判官意见修订三段输出（流式）。"""
        conf = self._decision_conf()
        recursion_limit = int(conf.get("recursion_limit", 25))
        
        payload = {
            "critic_review_json": review.model_dump_json(indent=2, ensure_ascii=False),
            "prior_speech_analysis": result.prior_speech_analysis or "",
            "identity_inference": result.identity_inference or "",
            "speech_suggestion": result.speech_suggestion or "",
        }
        user_content = self._revise_user_template.format(**payload)
        invoke_config = {"recursion_limit": recursion_limit}
        input_state = {"messages": [HumanMessage(content=user_content)]}
        phase_ctx = DecisionAgentContext(
            phase="revise",
            situation_sketch=situation_sketch,
            situation_sketch_narrative=situation_sketch_narrative,
        )

        messages: list[BaseMessage] | None = None
        try:
            async for event in self.agent.astream_events(
                input_state,
                invoke_config,
                version="v2",
                context=phase_ctx,
            ):
                et = event.get("event")
                if et == "on_chat_model_stream":
                    chunk = (event.get("data") or {}).get("chunk")
                    for kind, delta in stream_deltas_from_chunk(chunk):
                        yield {
                            "type": kind,
                            "delta": delta,
                            "iteration": iteration,
                            "phase": "revise",
                        }
                elif et == "on_tool_start":
                    name = (event.get("data") or {}).get("name") or event.get("name")
                    yield {
                        "type": "tool_start",
                        "name": str(name or ""),
                        "iteration": iteration,
                        "phase": "revise",
                    }
                elif et == "on_tool_end":
                    name = (event.get("data") or {}).get("name") or event.get("name")
                    yield {
                        "type": "tool_end",
                        "name": str(name or ""),
                        "iteration": iteration,
                        "phase": "revise",
                    }
                elif et == "on_chain_end":
                    out = (event.get("data") or {}).get("output")
                    if isinstance(out, dict) and out.get("messages") is not None:
                        messages = list(out["messages"])
        except Exception as e:
            logger.error(
                "DecisionAgent.revise_from_critic_stream: astream_events failed: %s",
                e,
            )
            raise RuntimeError(
                "决策修订流式失败（不支持非流式回退）。请检查模型是否支持流式。"
            ) from e

        if messages is None:
            raise RuntimeError(
                "决策修订流未返回最终 messages，请检查 LangGraph 与模型流式配置。"
            )

        raw = last_ai_text(messages)
        warnings = list(result.warnings)
        if not raw:
            warnings.append("修订阶段模型未返回可解析文本，保留上一轮输出")
        prior, identity, suggestion, parse_warnings = _parse_marked_output(raw or "")
        warnings.extend(parse_warnings)

        debug_prompt = _build_debug_prompt(
            self._revise_system, user_content, messages
        )

        dr = DecisionResult(
            prior_speech_analysis=prior or result.prior_speech_analysis,
            identity_inference=identity or result.identity_inference,
            speech_suggestion=suggestion or result.speech_suggestion,
            warnings=warnings,
            debug_prompt=debug_prompt,
            rule_hits=result.rule_hits,
            rule_critic_notes=result.rule_critic_notes,
            rule_critic_debug_prompt=result.rule_critic_debug_prompt,
        )
        yield {
            "type": "revise_complete",
            "result": dr.model_dump(),
            "iteration": iteration,
            "phase": "revise",
        }


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
