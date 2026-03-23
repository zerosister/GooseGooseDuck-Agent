import os

from backend.utils.config_handler import prompts_conf
from backend.utils.path_tool import get_abs_path
from backend.utils.logger_handler import logger


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] yaml config file doesn't have key {e}")
        raise e
    
    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] Failed to load rag prompts from {rag_prompt_path}")
        raise e
    
def load_summarize_prompt():
    """加载会议中每人发言总结 prompt，并从 data 注入角色目录与黑话。"""
    try:
        summarize_prompt_path = get_abs_path(prompts_conf['summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_summarize_prompt] yaml config file doesn't have key {e}")
        raise e

    try:
        text = open(summarize_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_summarize_prompt] Failed to load from {summarize_prompt_path}")
        raise e
    return text


def load_decision_prompts() -> tuple[str, str]:
    """加载决策 Agent 的 system 与 user 模板（user 侧含 {decision_context} 等占位符）。"""
    try:
        system_path = get_abs_path(prompts_conf["decision_agent_system_prompt_path"])
        user_path = get_abs_path(prompts_conf["decision_agent_user_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_decision_prompts] yaml config file doesn't have key {e}")
        raise e
    try:
        system_text = open(system_path, "r", encoding="utf-8").read()
        user_text = open(user_path, "r", encoding="utf-8").read()
    except OSError as e:
        logger.error(f"[load_decision_prompts] Failed to load: {e}")
        raise e
    return system_text, user_text