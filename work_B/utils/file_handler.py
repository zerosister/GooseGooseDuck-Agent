import hashlib
import os
from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

def get_file_md5_hex(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        return
    if not os.path.isfile(file_path):
        logger.error(f"Path {file_path} is not a file")
        return
    md5_obj = hashlib.md5()

    # 分段读取文件
    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:     #需要按照二进制读取
            while chunk:= f.read(chunk_size):
                md5_obj.update(chunk)
                """
                等价于
                chunk = f.read(chunk_size)
                while chunk:
                    md5_obj.update(chunk)
                    chunk = f.read(chunk_size)
                """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[get_file_md5_hex]Error while calculating md5 of file {file_path}: {e}")


def listdir_with_allowed_type(dir_path, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(dir_path):
        logger.error(f"[listdir_with_allowed_type]Path {dir_path} is not a directory")
        return allowed_types
    
    for f in os.listdir(dir_path):
        if f.endswith(allowed_types):
            files.append(os.path.join(dir_path, f))
    
    return tuple(files)
    

def pdf_loader(file_path: str, password: str = None) -> list[Document]:
    return PyPDFLoader(file_path=file_path, password=password).load()

def text_loader(file_path: str) -> list[Document]:
    return TextLoader(file_path=file_path, encoding="utf-8").load()
    

