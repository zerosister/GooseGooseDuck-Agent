import os
from typing import List, Dict

from dotenv import load_dotenv

from utils.logger import log_error, log_event
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class TongyiClient:
    def __init__(self, logger, model: str = "qwen3-32b") -> None:
        load_dotenv()
        self.logger = logger
        self.model = ChatTongyi(
            model=model,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个鹅鸭杀分析助手。根据以下发言列表，输出 JSON，以下为发言列表："),
            MessagesPlaceholder(variable_name="speeches"),
            ("user", "请根据发言列表，输出 JSON：{'players':[{'name':'玩家名','guess_role':'角色猜测','reason':'理由'}]}'"),
        ])

    async def infer_identities(self, session_id: str, speeches: List[Dict]) -> Dict:
        prompt = self.prompt.invoke({"speeches": speeches})
        log_event(
            self.logger,
            event_type="llm_request",
            session_id=session_id,
            payload={"model": self.model, "speech_count": len(speeches), "prompt": prompt},
        )
        
        try:
            log_event(
                self.logger,
                event_type="llm_stream_start",
                session_id=session_id,
                payload={"model": self.model, "speech_count": len(speeches), "prompt": prompt},
            )
            # 异步流式输出
            async for chunk in self.model.astream(prompt):
                delta = getattr(chunk, "content", None)
                if delta:
                    yield delta
        except Exception as exc:  # pragma: no cover
            log_error(
                self.logger,
                event_type="llm_exception",
                session_id=session_id,
                error=str(exc),
            )
            raise exc