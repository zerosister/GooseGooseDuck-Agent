import sys
from loguru import logger
from backend.utils.pathtool import get_abs_path

# 创建日志目录
LOG_DIR = get_abs_path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger():
    # 移除默认控制台输出
    logger.remove()

    # 添加自定义控制台输出 (带颜色)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        enqueue=True # 异步写入，不阻塞主逻辑
    )

    # 添加文件输出 (按天轮转，自动清理旧日志)
    logger.add(
        LOG_DIR / "ggd_agent_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="10 days",
        compression="zip",
        level="DEBUG",
        enqueue=True
    )

    return logger

# 初始化
log = setup_logger()