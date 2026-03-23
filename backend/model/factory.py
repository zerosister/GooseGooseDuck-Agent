from abc import ABC, abstractmethod
from typing import Optional
import threading

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from backend.utils.config_handler import rag_conf

_lock = threading.Lock()
_current_backend: str = "api"
_chat_model_api: Optional[BaseChatModel] = None
_chat_model_local: Optional[BaseChatModel] = None


def _init_api_model() -> BaseChatModel:
    global _chat_model_api
    if _chat_model_api is None:
        _chat_model_api = ChatTongyi(model=rag_conf['chat_model_name'])
    return _chat_model_api


def _init_local_model() -> BaseChatModel:
    global _chat_model_local
    if _chat_model_local is None:
        from langchain_ollama import ChatOllama
        _chat_model_local = ChatOllama(
            model="qwen2.5:7b-instruct",
            base_url="http://localhost:11434",
            num_ctx=4096,
        )
    return _chat_model_local


def get_chat_model() -> BaseChatModel:
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        return _init_api_model()


def set_model_backend(backend: str) -> str:
    global _current_backend
    if backend not in ("api", "local"):
        raise ValueError("backend must be 'api' or 'local'")
    with _lock:
        _current_backend = backend
    return _current_backend


def get_current_backend() -> str:
    return _current_backend


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])


chat_model = _init_api_model()
embedding_model = EmbeddingsFactory().generator()
