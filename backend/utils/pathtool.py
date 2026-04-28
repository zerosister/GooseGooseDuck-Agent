import os
from pathlib import Path

def get_root_path():
    # 使用 .resolve() 可以处理掉路径中的 .. 或软链接，更安全
    return Path(__file__).resolve().parent.parent.parent

def get_abs_path(name):
    # Path 对象自带拼接功能，用 / 比 os.path.join 更直观
    return get_root_path() / name