"""Backend services."""
from langchain_core.tools import tool

from backend.services.rag.rag_service import RagSummarizeService

rag = RagSummarizeService()


@tool(description="RAG 查询鹅鸭杀规则/词条库")
async def rag_query(query: str) -> str:
    return await rag.arag_summarize(query)
