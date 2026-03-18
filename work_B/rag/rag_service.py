"""
总结服务类。用户提问，搜索资料，获取答案。
"""
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.prompt_loader import load_rag_prompts
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def print_prompt(prompt_text):
    print("==="*20)
    print(prompt_text)
    return prompt_text

class RagSummarizeService(object):
    def __init__(self):
        self.vec_store = VectorStoreService()
        self.retriever = self.vec_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template =PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]：内容：{doc.page_content} | 参考元数据： {doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合什么样的扫地机器人？"))