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
    """加载会议中每人发言总结 prompt"""
    try:
        summarize_prompt_path = get_abs_path(prompts_conf['summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] yaml config file doesn't have key {e}")
        raise e
    
    try:
        return open(summarize_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] Failed to load rag prompts from {summarize_prompt_path}")
        raise e


def load_decision_prompt():
    try:
        path = get_abs_path(prompts_conf["decision_agent_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_decision_prompt] yaml config file doesn't have key {e}")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_decision_prompt] Failed to load from {path}")
        raise e