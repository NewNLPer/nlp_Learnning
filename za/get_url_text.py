# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/14 13:11
coding with comment！！！
"""
import requests
from bs4 import BeautifulSoup
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

save_path = r"C:\Users\NewNLPer\Desktop\school_rule.txt"

url_list =[
            'https://www.tsinghua.edu.cn/info/1094/82878.htm',
           "https://www.gov.cn/zhengce/zhengceku/2020-12/29/content_5574650.htm",
           "http://www.moe.gov.cn/jyb_xxgk/gk_gbgg/moe_0/moe_495/moe_1073/tnull_11916.html"
           ]

def get_text_from_url(url):
    # 发送HTTP请求并获取响应
    response = requests.get(url)
    # 检查请求是否成功
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage: {response.status_code}")
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 获取所有文本内容
    text = soup.get_text(separator='\n', strip=True)
    return text

if __name__ == "__main__":
    text = ''
    for url in tqdm(url_list):
        text += get_text_from_url(url)
    with open(save_path,"w",encoding="utf-8") as f:
        f.write(text)
    print("已将爬取的网页文本写入%s"%save_path)



