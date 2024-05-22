# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/21 17:41
coding with comment！！！
"""
import json
import time
import requests
from bs4 import BeautifulSoup
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# 景点知识库的保存路径
save_path = r"./knowledge_seed.json"

# 去哪儿官网的地级市景点信息列表
city_url_list = {
    "青岛": "https://travel.qunar.com/p-cs299783-qingdao-jingdian-3-1",
    "济南": "https://travel.qunar.com/p-cs300150-jinan-jingdian-3-1",
    "烟台": "https://travel.qunar.com/p-cs299824-yantai-jingdian-3-1",
    "泰安": "https://travel.qunar.com/p-cs300151-taian-jingdian-3-1",
    "威海": "https://travel.qunar.com/p-cs300115-weihai-jingdian-3-1",
    "潍坊": "https://travel.qunar.com/p-cs299800-weifang-jingdian-3-1",
    "临沂": "https://travel.qunar.com/p-cs300114-linyi-jingdian-3-1",
    "枣庄": "https://travel.qunar.com/p-cs300154-zaozhuang-jingdian-3-1",
    "聊城": "https://travel.qunar.com/p-cs299823-liaocheng-jingdian-3-1",
    "日照": "https://travel.qunar.com/p-cs300153-rizhao-jingdian-3-1"
}

def get_travel_info(text):
    """
    :param text: 爬取的网页文本 type==list
    :return:
    "景点*":{
                "景点名称": " ** ",
                "地点位置": " ** ",
                "景点简要介绍": " ** ",
                "景点详细介绍": " ** ",
                "建游览时间": " ** "
            }
    """
    area = None
    introduction = []
    travel_time = "2小时"
    intro = 0

    for i in range(len(text)):
        if text[i] == "驴友点评" and not intro:
            start = i + 4
            while start < len(text) and text[start] != "地址:":
                if text[start] not in ["展开全部","收起"]:
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


def extract_links(url):
    # 发送HTTP请求获取网页内容
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    # 找到所有的<a>标签
    a_tags = soup.find_all('a')
    # 提取href属性中的链接
    links = [a.get('href') for a in a_tags if a.get('href')]
    url_set = set()
    for link in links:
        if "p-oi" in link:
            url_set.add(link)
    return url_set


if __name__ == "__main__":
    errno_nums = 0
    knowlegde = {}
    for key in tqdm(city_url_list,"地级市遍历中 ... "):
        # qps太高，会导致ip被封禁
        time.sleep(5)
        song_url = extract_links(city_url_list[key])
        nums = 1
        knowlegde[key] = {}
        for item in tqdm(song_url,desc="各景点数据抽取中 ... "):
            # qps太高，会导致ip被封禁
            time.sleep(5)
            text = get_text_from_url(item)
            try:
                main_dic = get_travel_info(text)
            except:
                errno_nums += 1
                print("已出现{}条错误".format(errno_nums))
                continue

            knowlegde[key]["景点"+str(nums)] = main_dic
            nums += 1
    print("数据处理完毕，正在写入{}".format(save_path))
    with open(save_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(knowlegde,ensure_ascii=False))
    print("景点知识库已成功写入{}！".format(save_path))







