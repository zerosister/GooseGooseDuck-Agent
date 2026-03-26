from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from backend.utils.config_handler import model_conf

_lock = threading.Lock()
_current_backend: str = "api"
_rag_chat_model: Optional[BaseChatModel] = None
_agent_chat_model: Optional[BaseChatModel] = None
_decision_chat_model: Optional[BaseChatModel] = None
_decision_chat_model_stream: Optional[BaseChatModel] = None
_memory_chat_model: Optional[BaseChatModel] = None
_rule_critic_chat_model: Optional[BaseChatModel] = None
_rule_critic_chat_model_stream: Optional[BaseChatModel] = None
_chat_model_local: Optional[BaseChatModel] = None


def _api_key() -> str:
    k = model_conf.get("api_key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    return os.environ.get("DASHSCOPE_API_KEY", "") or ""


def _section_kwargs(section: str) -> dict[str, Any]:
    sec = model_conf.get(section)
    if not isinstance(sec, dict):
        return {"temperature": 0.7}
    out: dict[str, Any] = {}
    for key in ("enable_thinking", "incremental_output", "temperature"):
        if key in sec:
            out[key] = sec[key]
    if "temperature" not in out:
        out["temperature"] = 0.7
    return out


def _create_tongyi(model_name: str, section: str) -> ChatTongyi:
    return ChatTongyi(
        model=model_name,
        api_key=_api_key(),
        model_kwargs=_section_kwargs(section),
    )


def _decision_kwargs_for_invoke() -> dict[str, Any]:
    """通义 API：incremental_output 仅允许真正的流式调用；ainvoke/invoke 会报错，必须去掉。"""
    kw = _section_kwargs("decision")
    kw.pop("incremental_output", None)
    return kw


def _decision_kwargs_for_stream() -> dict[str, Any]:
    """流式专用（astream / astream_events）；可保留 incremental_output。"""
    return _section_kwargs("decision")


def _rule_critic_kwargs_for_stream() -> dict[str, Any]:
    """规则判官流式专用 astream_events；可保留 incremental_output。"""
    return _section_kwargs("rule_critic")


def _decision_model_name() -> str:
    d = model_conf.get("decision")
    if isinstance(d, dict):
        mn = d.get("model_name")
        if mn is not None and str(mn).strip():
            return str(mn).strip()
    return str(model_conf["agent_model_name"])


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


def get_rag_chat_model() -> BaseChatModel:
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _rag_chat_model
        if _rag_chat_model is None:
            _rag_chat_model = _create_tongyi(
                str(model_conf["rag_model_name"]), "rag"
            )
        return _rag_chat_model


def get_agent_chat_model() -> BaseChatModel:
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _agent_chat_model
        if _agent_chat_model is None:
            _agent_chat_model = _create_tongyi(
                str(model_conf["agent_model_name"]), "agent"
            )
        return _agent_chat_model


def get_decision_chat_model() -> BaseChatModel:
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _decision_chat_model
        if _decision_chat_model is None:
            _decision_chat_model = ChatTongyi(
                model=_decision_model_name(),
                api_key=_api_key(),
                model_kwargs=_decision_kwargs_for_invoke(),
            )
        return _decision_chat_model


def get_decision_chat_model_stream() -> BaseChatModel:
    """决策首轮流式专用：可含 incremental_output；勿用于 ainvoke。"""
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _decision_chat_model_stream
        if _decision_chat_model_stream is None:
            _decision_chat_model_stream = ChatTongyi(
                model=_decision_model_name(),
                api_key=_api_key(),
                model_kwargs=_decision_kwargs_for_stream(),
            )
        return _decision_chat_model_stream


def get_memory_chat_model() -> BaseChatModel:
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _memory_chat_model
        if _memory_chat_model is None:
            _memory_chat_model = _create_tongyi(
                str(model_conf["agent_model_name"]), "memory"
            )
        return _memory_chat_model


def get_rule_critic_chat_model() -> BaseChatModel:
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _rule_critic_chat_model
        if _rule_critic_chat_model is None:
            _rule_critic_chat_model = _create_tongyi(
                str(model_conf["agent_model_name"]), "rule_critic"
            )
        return _rule_critic_chat_model


def get_rule_critic_chat_model_stream() -> BaseChatModel:
    """规则判官决策评审流式专用；可含 incremental_output；勿用于 ainvoke。"""
    with _lock:
        if _current_backend == "local":
            return _init_local_model()
        global _rule_critic_chat_model_stream
        if _rule_critic_chat_model_stream is None:
            _rule_critic_chat_model_stream = ChatTongyi(
                model=str(model_conf["agent_model_name"]),
                api_key=_api_key(),
                model_kwargs=_rule_critic_kwargs_for_stream(),
            )
        return _rule_critic_chat_model_stream


def set_model_backend(backend: str) -> str:
    """切换后端时清空缓存并刷新模块级导出的 chat 实例（新 getter 调用会得到新模型）。"""
    global _current_backend
    global _rag_chat_model, _agent_chat_model, _decision_chat_model
    global _decision_chat_model_stream
    global _memory_chat_model, _rule_critic_chat_model, _rule_critic_chat_model_stream
    global _chat_model_local
    global rag_chat_model, agent_chat_model, decision_chat_model
    global decision_chat_model_stream
    global memory_chat_model, rule_critic_chat_model, rule_critic_chat_model_stream
    if backend not in ("api", "local"):
        raise ValueError("backend must be 'api' or 'local'")
    with _lock:
        _current_backend = backend
        _rag_chat_model = None
        _agent_chat_model = None
        _decision_chat_model = None
        _decision_chat_model_stream = None
        _memory_chat_model = None
        _rule_critic_chat_model = None
        _rule_critic_chat_model_stream = None
        _chat_model_local = None
    rag_chat_model = get_rag_chat_model()
    agent_chat_model = get_agent_chat_model()
    decision_chat_model = get_decision_chat_model()
    decision_chat_model_stream = get_decision_chat_model_stream()
    memory_chat_model = get_memory_chat_model()
    rule_critic_chat_model = get_rule_critic_chat_model()
    rule_critic_chat_model_stream = get_rule_critic_chat_model_stream()
    return _current_backend


def get_current_backend() -> str:
    return _current_backend


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=model_conf["embedding_model_name"])


rag_chat_model = get_rag_chat_model()
agent_chat_model = get_agent_chat_model()
decision_chat_model = get_decision_chat_model()
decision_chat_model_stream = get_decision_chat_model_stream()
memory_chat_model = get_memory_chat_model()
rule_critic_chat_model = get_rule_critic_chat_model()
rule_critic_chat_model_stream = get_rule_critic_chat_model_stream()
embedding_model = EmbeddingsFactory().generator()
