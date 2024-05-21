# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/21 17:41
coding with comment！！！
"""
import requests
from bs4 import BeautifulSoup
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

save_path = r"C:\Users\NewNLPer\Desktop\city_2.txt"

url_list =[
            "https://travel.qunar.com/p-oi710634-daimiao"
           ]


def get_travel_info(text):
    area = None
    introduction = []
    travel_time = "2小时"
    intro = 0

    for i in tqdm(range(len(text)), desc="Processing travel info..."):
        if text[i] == "驴友点评" and not intro:
            start = i + 4
            while start < len(text) and text[start] != "地址:":
                introduction.append(text[start])
                start += 1
            area = text[start + 1]
            intro = 1
        elif "建议游览时间：" in text[i]:
            travel_time = text[i].split("建议游览时间：")[-1]

    return {
        "景点名称":text[0][4:].split("-")[0],
        "地点位置": area,
        "景点详细介绍": introduction,
        "景点简要介绍":introduction[0],
        "建议游览时间": travel_time
    }

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
    return text.split("\n")

if __name__ == "__main__":
    text = get_text_from_url(url_list[0])
    print(get_travel_info(text))




