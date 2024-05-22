# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/22 15:45
coding with comment！！！
"""
import json
from text2vec import SentenceModel
import numpy as np
from scipy.spatial.distance import cosine
import requests
from zhipuai import ZhipuAI
import warnings
warnings.filterwarnings("ignore")


gaode_api_key = "8d8b55e388a180c9c7913e8f5e8ab10b"
zhipu_api_key = "042436202c1b74e82bda600ffb2d7c5d.SJgSFxMKmg0RgcZc"

class Sort_dic():
    def __init__(self,nums):
        self.dic = {}
        self.nums = nums

    def add_item(self, key, value):
        # 如果字典项数已达到最大值，删除最小值的项
        if len(self.dic) >= self.nums:
            min_key = min(self.dic, key=self.dic.get)
            if self.dic[min_key] < value:
                del self.dic[min_key]
                self.dic[key] = value
        else:
            self.dic[key] = value

    def get_similar_text(self):
        self.dic = dict(sorted(self.dic.items(), key=lambda x: x[1],reverse = True))
        return self.dic

class My_langchain():
    def __init__(self,knowledge_file_path,em_model_path):
        # 景点知识库文件地址
        self.knowledge_file = knowledge_file_path
        # 词嵌入矩阵模型权重地址
        self.em_model_path = em_model_path
        with open(self.knowledge_file,"r",encoding="utf-8") as f:
            self.knowledge = json.load(f)
        self.em_model = SentenceModel(em_model_path)

    def len_limit(self,text):
        """
        :param text:景点简要介绍
        :return: 输出更加简短的景点简要介绍
        """
        if len(text) > 128:
            text = text.split("。")[0]
            return text
        return text

    def get_cos_sm(self,text1, text2):
        """
        计算文本cos相似度
        :param text1:旅游意愿
        :param text2:景点简要介绍
        :return:输出相似度
        """
        em_1 = self.em_model.encode(text1)
        em_2 = self.em_model.encode(text2)
        text1 = np.array(em_1)
        text2 = np.array(em_2)
        similarity = 1 - cosine(text1, text2)
        return similarity

    def get_some_sight(self,city,traval_goo,k):
        """
        :param city:目的地
        :param traval_goo:旅游意图
        :return: 目的地与旅游意图相符的k个景点
        """
        sight = []
        if not traval_goo:# 如果没有旅行意图，仅检索前k，因为爬取的时候是按照人气爬取的，所以合理。
            for item in self.knowledge[city]:
                intro = "景点名称：{}，地点位置：{}，景点介绍：{}，建议游览时间：{}".format(
                    self.knowledge[city][item]["景点名称"],
                    self.knowledge[city][item]["地点位置"],
                    self.len_limit(self.knowledge[city][item]["景点简要介绍"]),
                    self.knowledge[city][item]["建议游览时间"]
                )
                sight.append(intro)
                if len(sight) == k:
                    break
        else:# 若有意图将意图与景点的简要介绍计算相似度，取相似度最高的前k个。
            recall_sm = Sort_dic(k)
            for item in self.knowledge[city]:
                simi = self.get_cos_sm(traval_goo,self.knowledge[city][item]["景点简要介绍"])
                recall_sm.add_item(item,simi)
            recall_sm_out = recall_sm.get_similar_text()
            for key in recall_sm_out:
                intro = "景点名称：{}，地点位置：{}，景点介绍：{}，建议游览时间：{}".format(
                    self.knowledge[city][key]["景点名称"],
                    self.knowledge[city][key]["地点位置"],
                    self.len_limit(self.knowledge[city][key]["景点简要介绍"]),
                    self.knowledge[city][key]["建议游览时间"]
                )
                sight.append(intro)

        return sight


def get_completion(prompt):
    client = ZhipuAI(api_key=zhipu_api_key)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

def get_lon_lat(i):
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    parameters = {
        'key':gaode_api_key,
        'address':'%s' % i
        }
    page_resource = requests.get(url,params=parameters)
    text = page_resource.text
    data = json.loads(text)
    lon_lat = data["geocodes"][0]['location']
    return lon_lat

def routes(origin,destination):
    origin_1  = get_lon_lat(origin)
    destination_1 = get_lon_lat(destination)
    parameters = {'key':gaode_api_key,'origin':origin_1,'destination':destination_1}
    response = requests.get('https://restapi.amap.com/v3/direction/driving?parameters',params=parameters)
    text = json.loads(response.text)
    duration = text['route']['paths'][0]['duration']
    return "从{}到{}驾车需要{}个小时。".format(origin, destination, str(round(int(duration)/3600,2)))


def get_Final_Prompt(sight,traval_time,question):
    promot = """
    下面是一些景点信息和行程消耗时间，请根据用户的需求帮其指定一个旅游的行程规划。
    用户需求：{}
    路程时间：{}
    目的地相关景点信息：{}
    """
    return promot.format(question,traval_time,"\n".join(sight))

def main():
    recall_k = 8
    knowledge_file_path = r"C:\Users\NewNLPer\Desktop\knowledge_seed.json"
    em_model_path = "D:\google下载"

    question = input("question：")

    prompt = """
    给定一个问题，请帮我对其进行信息抽取，仅需输出抽取后的结果即可.请严格按照给定的格式进行信息抽取，抽取信息后的输出格式为：
    {{
        "出发地": "**/未提及",
        "目的地": "[**]/[未提及]",
        "旅游意愿": "**/未提及"
    }}
    下面给出一个样例；
    quesetion = " 我目前在日照市，我想去威海看海，请帮我规划一个三天的行程。"
    {{
        "出发地": "日照",
        "目的地": "威海",
        "旅游意愿": "看海"
    }}
    请根据上述样例，对下面的问题进行信息抽取，仅需输出抽取后的结果即可。
    new_question = "{}"
    """

    prompt = prompt.format(question)
    extral_inf = json.loads(get_completion(prompt))
    gs = My_langchain(knowledge_file_path,em_model_path)
    sight = gs.get_some_sight(extral_inf["目的地"],extral_inf["旅游意愿"],recall_k)
    traval_time = routes(extral_inf["出发地"],extral_inf["目的地"])
    finall_prompt = get_Final_Prompt(sight,traval_time,question)
    # print(finall_prompt)
    answer = get_Final_Prompt(question)
    answer_rag = get_completion(finall_prompt)
    print(answer)
    print('==================================================')
    print(answer_rag)




if __name__ == "__main__":
    main()

"""
我目前在烟台市，我想去日照看海，请帮我规划一个两天的行程。
"""
