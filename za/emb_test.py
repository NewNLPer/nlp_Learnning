# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/14 15:01
coding with comment！！！
"""

import warnings
warnings.filterwarnings("ignore")
from text2vec import SentenceModel

txt_path = r"C:\Users\NewNLPer\Desktop\school_rule_pre_2.txt"

em_model_path = "D:\google下载"

if __name__ == "__main__":
    t2v_model = SentenceModel(em_model_path)
    print(t2v_model)
    with open(txt_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for item in lines:
            em = t2v_model.encode(item)
            print("文本为：%s"%(item))
            print("其对应的词嵌入向量为：%s"%(em))
            print("词嵌入向量的长度为：%s"%(len(em)))
            input()






