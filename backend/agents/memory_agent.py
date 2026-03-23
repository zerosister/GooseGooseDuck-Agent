from backend.schemas.contract import IngestionOutput, MemorySummary, PlayerState
from datetime import datetime
from backend.utils.config_handler import agent_conf
from backend.model.factory import chat_model
from backend.utils.prompt_loader import load_summarize_prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MemoryAgent:
    def __init__(self):
        self.prompt_text = load_summarize_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chain = self.prompt_template | chat_model | StrOutputParser()
        
    async def process(self, ingestions: list[IngestionOutput], summary: MemorySummary, recent_n: int = agent_conf['memory_agent']['recent_n']) -> MemorySummary:
        """
        为每个玩家生成总结
        """

        # 2. 为每个玩家生成画像
        ingestion_text = ingestions[-1].model_dump_json(indent=2, ensure_ascii=False)
        summary_text = self.chain.invoke({"ingestion": ingestion_text})
        player_state = PlayerState(
            speaker_id=ingestions[-1].metadata.get("speaker_id"),
            latest_stance=summary_text,
            emotion_trend="稳定"
        )
        summary.player_summaries.append(player_state)
        return summary
        