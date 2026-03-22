from backend.schemas.contract import IngestionOutput, MemorySummary, PlayerState
from datetime import datetime
from utils.config_handler import agent_conf
from model.factory import chat_model
from utils.prompt_loader import load_summarize_prompt
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
        # 1. 按玩家分组
        # players_data = {}
        # if recent_n > len(ingestions):
        #     recent_n = len(ingestions)
        # for ing in ingestions[-recent_n:]:
        #     uid = ing.metadata.get("speaker_id")
        #     if uid not in players_data:
        #         players_data[uid] = []
        #     players_data[uid].append(ing)

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
        
        # player_summaries = []
        # for uid, logs in players_data.items():
        #     # 逻辑示例：取该玩家最后一条发言作为最新立场
        #     latest_msg = logs[-1].content
        #     # 提取所有指控...
        #     summary = PlayerState(
        #         speaker_id=uid,
        #         latest_stance=latest_msg,
        #         emotion_trend="稳定" # 占位逻辑
        #     )
        #     player_summaries.append(summary)

        # return MemorySummary(
        #     session_id=ingestions[0].session_id if ingestions else "unknown",
        #     player_summaries=player_summaries,
        #     timestamp=datetime.now().isoformat()
        # )