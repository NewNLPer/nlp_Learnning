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

gaode_api_key = "8d8b55e388a180c9c7913e8f5e8ab10b"

class Sort_dic():
    """
    定义一类具有容量要求的排序字典，方便召回最相关的文段
    """
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

        self.knowledge_file = knowledge_file_path
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
        if not traval_goo:
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
        else:
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
    origin  = get_lon_lat(origin)
    destination = get_lon_lat(destination)
    parameters = {'key':gaode_api_key,'origin':origin,'destination':destination}
    response = requests.get('https://restapi.amap.com/v3/direction/driving?parameters',params=parameters)
    text = json.loads(response.text)
    duration = text['route']['paths'][0]['duration']
    # 换算成小时
    return str(round(int(duration)/3600,2)) + "h"




if __name__ == "__main__":

    knowledge_file_path = r"C:\Users\NewNLPer\Desktop\knowledge_seed.json"

    em_model_path = "D:\google下载"

    sf = My_langchain(knowledge_file_path, em_model_path)

    res = sf.get_some_sight("日照","去看海，感受大海的潮来潮去",5)

    print("\n".join(res))