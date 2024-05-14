# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/14 13:11
coding with comment！！！
"""
import requests
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore")

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

# 示例URL
url = 'https://www.gov.cn/zhengce/zhengceku/2020-12/29/content_5574650.htm'
text = get_text_from_url(url)
print(type(text))
print(text)
