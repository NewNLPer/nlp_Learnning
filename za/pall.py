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
    text1 = np.array(text1_em)
    text2 = np.array(text2_em)
    similarity = 1 - cosine(text1, text2)
    return similarity


if __name__ == "__main__":
    question = "考试可以作弊嘛？"

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

            similar_text.add_item(item,simi)

        similar_text.get_similar_text()








