# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/14 15:30
coding with comment！！！
"""


from scipy.spatial.distance import cosine
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from text2vec import SentenceModel
from tqdm import tqdm

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
        self.dic = dict(sorted(self.dic.items(), key=lambda x: x[1]))
        for item in self.dic:
            print("相似度为：%s"%(self.dic[item]))
            print("对应句为：%s"%(item))
            print('================================================')

def get_cos_sm(text1_em,text2_em):
    """
    计算文本cos相似度
    :param text1_em:
    :param text2_em:
    :return:
    """
    text1 = np.array(text1_em)
    text2 = np.array(text2_em)
    similarity = 1 - cosine(text1, text2)
    return similarity

if __name__ == "__main__":

    question = "学生考试可以作弊嘛？"
    # 定义一个可以存储3个值的排序字典，key为字段，value为cos相似度
    similar_text = Sort_dic(3)
    em_model_path = "D:\google下载"
    txt_path = r"C:\Users\NewNLPer\Desktop\school_rule_pre_2.txt"
    # 加载词嵌入权重
    t2v_model = SentenceModel(em_model_path)
    # 计算question的词嵌入矩阵
    oral_em = em = t2v_model.encode(question)
    with open(txt_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        # 遍历知识库里的每个字段
        for item in tqdm(lines):
            # 计算知识库里每个字段的词嵌入矩阵
            em = t2v_model.encode(item)
            # 计算cos相似度
            simi = get_cos_sm(oral_em,em)
            # 将cos相似度值大的放进上述定义的容器里
            similar_text.add_item(item,simi)
        # 打印容器里的值
        similar_text.get_similar_text()








