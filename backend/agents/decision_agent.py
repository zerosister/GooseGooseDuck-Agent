from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from backend.schemas.contract import IngestionOutput, MemorySummary
from backend.schemas.decision import DecisionContext, DecisionResult
from backend.model.factory import chat_model

if TYPE_CHECKING:
    from backend.services.rag.rag_service import RagSummarizeService
from backend.utils.config_handler import agent_conf
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import load_decision_prompt


def _build_rag_queries(ctx: DecisionContext, max_n: int) -> list[str]:
    queries: list[str] = []
    queries.append(f"鹅鸭杀 {ctx.role_name} 角色技能与规则要点")
    al = ctx.alignment
    if isinstance(al, str) and al and al != "unknown":
        queries.append(f"鹅鸭杀 {al} 阵营 胜利条件与基本玩法")
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= max_n:
            break
    return out[:max_n]


def _parse_marked_output(text: str) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    m_prior = "【前序发言分析】"
    m_sug = "【发言建议】"
    if m_prior not in text or m_sug not in text:
        warnings.append("模型未按标记格式输出，已回退为全文归入分析段")
        return text.strip(), "", warnings
    _, rest = text.split(m_prior, 1)
    if m_sug in rest:
        prior, _, sug = rest.partition(m_sug)
        return prior.strip(), sug.strip(), warnings
    warnings.append("缺少【发言建议】标记")
    return rest.strip(), "", warnings


class DecisionAgent:
    """
    规则侧 RAG：默认通过 RagSummarizeService 做「检索 + LLM 摘要」；
    可将 `use_rag_summarize` 设为 false 以仅使用向量库原文摘录。
    """

    def __init__(
        self,
        rag: Optional["RagSummarizeService"] = None,
        model=None,
    ):
        if rag is None:
            from backend.services.rag.rag_service import RagSummarizeService

            self._rag = RagSummarizeService()
        else:
            self._rag = rag
        self.model = model or chat_model
        self._prompt_text = load_decision_prompt()
        self._template = PromptTemplate.from_template(self._prompt_text)
        self._chain = self._template | self.model | StrOutputParser()
        logger.info("DecisionAgent initialized (RagSummarizeService + decision chain)")

    def _decision_conf(self) -> dict:
        return agent_conf.get("decision_agent", {})

    async def _retrieve_docs(self, query: str) -> list[Document]:
        try:
            return await self._rag.retriever.ainvoke(query)
        except (AttributeError, NotImplementedError, TypeError):
            return await asyncio.to_thread(self._rag.retriever.invoke, query)

    @staticmethod
    def _format_rag_excerpt(query: str, docs: list[Document]) -> str:
        if not docs:
            return f"查询: {query}\n摘录: （无命中）"
        lines: list[str] = []
        for i, doc in enumerate(docs, 1):
            lines.append(
                f"[参考资料{i}] 内容：{doc.page_content} | 元数据：{doc.metadata}"
            )
        return f"查询: {query}\n摘录:\n" + "\n".join(lines)

    async def run(
        self,
        ctx: DecisionContext,
        memory: MemorySummary,
        ingestions: list[IngestionOutput] | None = None,
    ) -> DecisionResult:
        conf = self._decision_conf()
        recent_n = int(conf.get("recent_ingestion_n", 20))
        max_q = int(conf.get("max_rag_queries", 2))
        use_rag_summarize = bool(conf.get("use_rag_summarize", True))

        warnings: list[str] = []
        queries = _build_rag_queries(ctx, max_q)
        rag_queries_used = list(queries)

        excerpt_parts: list[str] = []
        for q in queries:
            try:
                if use_rag_summarize:
                    summary = await self._rag.arag_summarize(q)
                    excerpt_parts.append(f"查询: {q}\n规则摘要:\n{summary.strip()}")
                else:
                    docs = await self._retrieve_docs(q)
                    excerpt_parts.append(self._format_rag_excerpt(q, docs))
            except Exception as e:
                logger.error("rule_library RAG 失败: %s", q, exc_info=True)
                warnings.append(f"规则 RAG 失败: {q}: {e!s}")
                label = "规则摘要" if use_rag_summarize else "摘录"
                excerpt_parts.append(f"查询: {q}\n{label}: （失败）")

        rag_excerpts = "\n\n---\n\n".join(excerpt_parts) if excerpt_parts else "（未发起检索）"

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
            "rag_excerpts": rag_excerpts,
        }
        logger.info("DecisionAgent payload: %s", payload)

        rendered_prompt = self._template.format(**payload)

        try:
            raw = await self._chain.ainvoke(payload)
        except (AttributeError, NotImplementedError, TypeError):
            raw = await asyncio.to_thread(self._chain.invoke, payload)

        prior, suggestion, parse_warnings = _parse_marked_output(raw)
        warnings.extend(parse_warnings)

        return DecisionResult(
            prior_speech_analysis=prior,
            speech_suggestion=suggestion,
            rag_queries_used=rag_queries_used,
            warnings=warnings,
            debug_prompt=rendered_prompt,
        )


if __name__ == "__main__":
    sample = """【前序发言分析】
a
【发言建议】
b
"""
    p, s, w = _parse_marked_output(sample)
    print("parse:", p, s, w)
