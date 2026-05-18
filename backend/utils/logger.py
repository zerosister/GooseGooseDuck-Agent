import sys
import datetime
from loguru import logger
from backend.utils.pathtool import get_abs_path

# 创建日志目录
LOG_DIR = get_abs_path("logs")
LOG_DIR.mkdir(exist_ok=True)

def clean_old_logs_on_startup(days=7):
    """
    程序启动时显式检查，清理 7 天前的旧日志
    """
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=days)
    
    # 计数器，让控制台输出更干净
    deleted_count = 0
    
    for file_path in LOG_DIR.iterdir():
        if file_path.is_file():
            file_mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime < cutoff:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception:
                    pass # 启动时静默失败，不影响主程序运行
                    
    if deleted_count > 0:
        print(f"[Startup Clean] 成功清理了 {deleted_count} 个 7 天前的历史日志文件。")

def setup_logger():
    logger.remove()

    # 控制台输出 (INFO 级别)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        enqueue=True
    )

    # 文件输出 (DEBUG 级别，严格限制 7 天)
    logger.add(
        LOG_DIR / "ggd_agent_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",  # 长期运行时的自动清理策略
        compression="zip",
        level="DEBUG",
        enqueue=True
    )

    return logger

# 1. 每次重启都会秒级扫描文件夹，无感清理
clean_old_logs_on_startup(days=7)

# 2. 初始化日志
log = setup_logger()