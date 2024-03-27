# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/27 16:37
coding with comment！！！
"""
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
import langchain
from langchain.cache import InMemoryCache
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
from langchain.chains import RetrievalQA
import warnings
import os
os.environ["BAICHUAN_API_KEY"] = "sk-df4297f3987923adcb5d6f60cd746b79"
from langchain_community.llms import BaichuanLLM

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
# 启动llm的缓存
langchain.llm_cache = InMemoryCache()


loader = TextLoader(r"C:/Users/NewNLPer/Desktop/za/some_test/school_rule.txt",encoding = "utf-8")
data = loader.load()


# 初始化加载器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=52, chunk_overlap=26)

# 切割加载的 document
split_docs = text_splitter.split_documents(data)

EMBEDDING_MODEL = "D:\google下载"

# 考虑embedding
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device = 'cuda')

# 将切分好的文本片段转换为向量，并存入FAISS中：
db = FAISS.from_documents(split_docs, embeddings)
db.save_local(r"C:\Users\NewNLPer\Desktop\za\some_test\faiss") # 指定Faiss的位置

#  db = FAISS.load_local("/data/ubuntu/jijiezhou/data/faiss/",embeddings=embeddings)
db = FAISS.load_local(r"C:\Users\NewNLPer\Desktop\za\some_test\faiss", embeddings=embeddings, allow_dangerous_deserialization=True)

llm = BaichuanLLM()


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
        answer = qa.run((question+","+"请用中文来回答相关问题。"))
        print("BaichuanLLM的回答：" + answer)

"""
学生能抄作业嘛？
"""