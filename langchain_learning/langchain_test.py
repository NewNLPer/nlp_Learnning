# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/27 15:54
coding with comment！！！
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import logging
import requests
from typing import Optional, List
import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
# 启动llm的缓存
langchain.llm_cache = InMemoryCache()


class ChatGLM(LLM):
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        data = {'text': prompt}
        url = "http://10.1.0.68:1022/chat/"
        response = requests.post(url, json=data)
        if response.status_code != 200:
            return "error"
        resp = response.json()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, resp['result']]]
        return resp['result']


loader = TextLoader(r"/data/ubuntu/jijiezhou/data/school_rule.txt")
data = loader.load()

# 初始化加载器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=52, chunk_overlap=26)

# 切割加载的 document
split_docs = text_splitter.split_documents(data)

EMBEDDING_MODEL = "/data/ubuntu/public/models/embedding_pm/"

# 考虑embedding
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device = 'cuda')

# 将切分好的文本片段转换为向量，并存入FAISS中：
db = FAISS.from_documents(split_docs, embeddings)
db.save_local("/data/ubuntu/jijiezhou/data/faiss/") # 指定Faiss的位置

#  db = FAISS.load_local("/data/ubuntu/jijiezhou/data/faiss/",embeddings=embeddings)
db = FAISS.load_local("/data/ubuntu/jijiezhou/data/faiss/", embeddings=embeddings, allow_dangerous_deserialization=True)

llm = ChatGLM()
if __name__ == "__main__":
    while True:

        question = input()
        similarDocs = db.similarity_search(question, include_metadata = True, k = 3)
        print("--- 用户所提问题与外部知识库匹配度最高的三个句子如下； ---")
        print("=========================================================")
        for item in similarDocs:
            print(item.page_content)
            print("=========================================================")


        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever)
        answer = qa.run((question))
        print("ChatGLM的回答：" + answer)