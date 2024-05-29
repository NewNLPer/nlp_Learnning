# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/28 19:51
coding with comment！！！
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import logging
import langchain
from langchain.cache import InMemoryCache
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
import warnings
warnings.filterwarnings("ignore")
import gradio as gr
import argparse
from PyPDF2 import PdfMerger
from tqdm import tqdm
from collections import Counter
from zhipuai import ZhipuAI
import jieba

parser = argparse.ArgumentParser()
parser.add_argument('--baichuan_api_keys', type=str,default="",help='get baichuan api-keys')
parser.add_argument('--knowledge_file_path_list', nargs="+", default=[], help='add pdf_pdf path to list .')
parser.add_argument('--em_model_path', type=str, default="", help='form hf download model para')
parser.add_argument('--fassi_save_path', type=str, default="", help='once save path ,notice need creat a new dic to save')
parser.add_argument('--url_setting', type=str, default="127.0.0.1", help='setting url')
parser.add_argument('--pdf_combine_path', type=str, default="", help='after using will be deleted')
parser.add_argument('--n_gram', type=int, default=1, help='after using will be deleted')

args = parser.parse_args()

os.environ["BAICHUAN_API_KEY"] = args.baichuan_api_keys
print(args.knowledge_file_path_list)

def merge_pdfs(pdf_list, output_path):
    merger = PdfMerger()
    for pdf in tqdm(pdf_list,desc="Combine pdf file ... "):
        merger.append(pdf)
    merger.write(output_path)
    merger.close()

def has_files_in_directory(directory_path):
    # 检查路径是否存在且为目录
    if not os.path.exists(directory_path):
        return False
    if not os.path.isdir(directory_path):
        return False

    # 检查目录是否有文件
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            return True
    return False


def acc_rec_score(candidate, reference):
    candidate_tokens = jieba.lcut(candidate)
    reference_tokens = jieba.lcut(reference)
    n = args.n_gram
    # 计算N-gram频次
    candidate_ngrams = [tuple(candidate_tokens[i:i+n]) for i in range(len(candidate_tokens)-n+1)]
    reference_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens)-n+1)]
    count = sum(1 for item in candidate_ngrams if item in reference_ngrams)
    candidate_freq = Counter(candidate_ngrams)
    reference_freq = Counter(reference_ngrams)

    # 计算匹配的N-gram数量
    matches = sum(min(candidate_freq[ngram], reference_freq[ngram]) for ngram in candidate_freq)

    # 计算修正后的N-gram召回率
    total_reference_ngrams = len(reference_ngrams)
    recall = matches / total_reference_ngrams
    accuracy = count / len(candidate_ngrams)
    return [accuracy,recall]

moc = has_files_in_directory(args.fassi_save_path)

if moc:
    print("知识库emb矩阵已存在，无需重新生成 ...")
    embeddings = HuggingFaceEmbeddings(model_name=args.em_model_path)
    db = FAISS.load_local(args.fassi_save_path,
                          embeddings=embeddings,
                          allow_dangerous_deserialization=True)

else:
    print("知识库emb矩阵未存在，需重新生成，请耐心等待 ...")
    merge_pdfs(args.knowledge_file_path_list,args.pdf_combine_path)
    logging.basicConfig(level=logging.INFO)
    langchain.llm_cache = InMemoryCache()

    loader = PyPDFLoader(args.pdf_combine_path)

    data = loader.load()
    # 初始化加载器
    # # 初始化加载器,文本分片器将该txt文档切分为了每段有128个tokens，片段与片段之间有32个Tokens重叠的文本小片段
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(data)
    # 考虑embedding
    embeddings = HuggingFaceEmbeddings(model_name=args.em_model_path)
    embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device = 'cuda')
    # 将切分好的文本片段转换为向量，并存入FAISS中：
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(args.fassi_save_path) # 指定Faiss的位置
    db = FAISS.load_local(args.fassi_save_path,
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True)
    print("生成完毕，加载完毕！")
def get_completion(input):
    client = ZhipuAI(api_key=args.baichuan_api_keys)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": input},
        ],
    )
    return response.choices[0].message.content


def get_comletion(question):

    mark = "=========================================="

    text1_result = ""

    similarDocs = db.similarity_search(question, include_metadata=True, k=3)

    for item in similarDocs:
        text1_result += item.page_content
        text1_result += "\n"
        text1_result += mark
    text2_result = get_completion(question)


    prompt = """
            给定一个问题和一段学习材料，请根据所提供的学习材料来回答这个问题。
            问题：{}
            学习材料：{} """.format(question,text1_result)

    text3_answer = get_completion(prompt)

    text4_accuracy,text5_recall = acc_rec_score(text3_answer,text2_result)

    colored_text1 = ("<div style='margin-bottom: 20px;'>"
                     "<div style='border: 2px solid black; padding: 10px;'>"
                     "<div style='font-weight: bold; color:black; margin-bottom: 5px;'>Most Relevant to Question</div>"
                     "<div style='color:red; font-family:Times New Roman; font-size:16px;'>{}</div>"
                     "</div></div>").format(text1_result)

    colored_text2 = ("<div style='margin-bottom: 20px;'>"
                     "<div style='border: 2px solid black; padding: 10px;'>"
                     "<div style='font-weight: bold; color:black; margin-bottom: 5px;'>Baichuan Without RAG</div>"
                     "<div style='color:blue; font-family:Helvetica; font-size:16px;'>{}</div>"
                     "</div></div>").format(text2_result)

    colored_text3 = ("<div style='margin-bottom: 20px;'>"
                     "<div style='border: 2px solid black; padding: 10px;'>"
                     "<div style='font-weight: bold; color:black; margin-bottom: 5px;'>Baichuan With RAG</div>"
                     "<div style='color:green; font-family:Courier New; font-size:16px;'>{}</div>"
                     "</div></div>").format(text3_answer)

    colored_text4 = ("<div style='margin-bottom: 20px;'>"
                     "<div style='border: 2px solid black; padding: 10px;'>"
                     "<div style='font-weight: bold; color:black; margin-bottom: 5px;'>Accuracy</div>"
                     "<div style='color:purple; font-family:Courier New; font-size:16px;'>{}</div>"
                     "</div></div>").format(text4_accuracy)

    colored_text5 = ("<div style='margin-bottom: 20px;'>"
                     "<div style='border: 2px solid black; padding: 10px;'>"
                     "<div style='font-weight: bold; color:black; margin-bottom: 5px;'>Recall</div>"
                     "<div style='color:orange; font-family:Courier New; font-size:16px;'>{}</div>"
                     "</div></div>").format(text5_recall)

    # 使用flexbox布局将三个文本在一列中显示
    flex_container_style = "display:flex; flex-direction:column; align-items:flex-start;"
    combined_text = colored_text1 + colored_text2 + colored_text3 + colored_text4 + colored_text5

    return "<div style='{}'>{}</div>".format(flex_container_style, combined_text)



if __name__ == "__main__":

    demo = gr.Interface(
            fn=get_comletion,
            inputs="text",
            outputs="html",
            title="基于Langchain的计算机知识问答系统",
            description="欢迎使用！"
        )

    demo.launch(server_name = args.url_setting,server_port = 5910)
