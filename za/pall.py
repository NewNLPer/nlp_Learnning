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

    def __init__(self,nums):
        self.dic = {}
        self.nums = nums
    def del_(self):
        dic_s = dict(sorted(self.dic.items(), key=lambda item: item[1]))
        key = next(iter(dic_s))
        dic_s.pop(key)
        return dic_s

    def add_item(self, key, value):
        if len(self.dic) >= self.nums:
            self.dic = self.del_()
        self.dic[key] = value

    def get_similar_text(self):
        self.dic = dict(sorted(self.dic.items(), key=lambda item: item[1]))
        for key in self.dic:
            print(key)
            print(self.dic[key])


def get_cos_sm(text1_em,text2_em):
    text1 = np.array(text1_em)
    text2 = np.array(text2_em)
    similarity = 1 - cosine(text1, text2)
    return similarity


if __name__ == "__main__":
    question = "学生能不能逃课？"

    similar_text = Sort_dic(5)

    em_model_path = "D:\google下载"

    txt_path = r"C:\Users\NewNLPer\Desktop\school_rule.txt"

    t2v_model = SentenceModel(em_model_path)

    oral_em = em = t2v_model.encode(question)

    with open(txt_path,"r",encoding="utf-8") as f:

        lines = f.readlines()

        for item in tqdm(lines):

            em = t2v_model.encode(item)

            simi = get_cos_sm(oral_em,em)

            similar_text.add_item(simi,item)

    similar_text.get_similar_text()







