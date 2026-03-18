import os
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, text_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document

class VectorStoreService(object):
    def __init__(self, embedding=embedding_model):
        """embedding:嵌入模型的传入"""
        self.embedding = embedding

        self.vec_store = Chroma(
            collection_name=chroma_conf['collection_name'],
            embedding_function=embedding_model,
            persist_directory=chroma_conf['persist_directory']
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'],
            chunk_overlap=chroma_conf['chunk_overlap'],
            separators=chroma_conf['separators']
        )

    def get_retriever(self):
        return self.vec_store.as_retriever(search_kwargs={"k": chroma_conf['k']})
    
    def load_document(self):
        """
        读取文件，存入向量库，需要md5去重
        """

        def check_md5_hex(md5_hex: str):
            if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):
                # 创建文件
                open(get_abs_path(chroma_conf['md5_hex_store']), 'w', encoding="utf-8").close()
                return False
            
            with open(get_abs_path(chroma_conf['md5_hex_store']), 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_hex:
                        return True
            return False
        
        def save_md5_hex(md5_hex: str):
            with open(get_abs_path(chroma_conf['md5_hex_store']), 'a', encoding="utf-8") as f:
                f.write(md5_hex + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith('.pdf'):
                return pdf_loader(read_path)
            if read_path.endswith('.txt'):
                return text_loader(read_path)
            return []
        
        allowed_files_paths = listdir_with_allowed_type(
            chroma_conf['data_path'],
            tuple(chroma_conf['allow_knowledge_files_type'])
        )

        for path in allowed_files_paths:
            # 获取文件 MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"Document {path} has been loaded.")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"Document {path} is empty.")
                    continue

                split_document: list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.warning(f"Document {path} is empty after split.")
                    continue

                # 存入向量库
                self.vec_store.add_documents(split_document)
                save_md5_hex(md5_hex)

                logger.info(f"Document {path} has been loaded.")
            except Exception as e:
                # exc_info 为 True 会记录详细报错
                logger.error(f"Load document {path} failed.", exc_info=True)
                continue

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)

