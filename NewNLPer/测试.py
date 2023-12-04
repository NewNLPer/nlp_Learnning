"""
@author: NewNLPer
@time: 2023/1/13 18:57
coding with comment！！！
"""
import jieba
import pandas as pd
import re
from torchtext import vocab
from torchtext.data import Field

def chinese_pre(text_data):
    stop_words = pd.read_csv("D:/charm/pytorch数据/程序/programs/data/chap7/cnews/中文停用词库.txt", sep="\t",
                             header=None, names=["text"])
    # text_data = text_data.lower()    ## 字母转化为小写
    text_data = re.sub("\d+", "", text_data)   ## 去除数字，正则表达是的强大之处
    text_data = list(jieba.cut(text_data,cut_all=False))      ## 分词,使用精确模式(值得注意的是，jieba自动采用HMM来进行计算最大概率，从而实现分词)
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]     ## 去停用词和多余空格
    text_data = " ".join(text_data) ## 处理后的词语使用空格连接为字符串
    text_data=text_data.split(' ')
    return text_data



#
# dict_list={"体育": 0,"娱乐": 1,"家居": 2,"房产": 3,"教育": 4,
#             "时尚": 5,"时政": 6,"游戏": 7,"科技": 8,"财经": 9}
# mydict_new=dict(zip(dict_list.values(),dict_list.keys()))
# print(mydict_new)



'''
        1.21-----1.31
1.算法项目的学习(GRU,PINN)
2.sql项目
3.算法(leetcode,sql)(二分，排序，二叉树，链表，动态规划，)
4.八股文
5.面试问题搜集与准备
'''

'''
1.NLP资料阅读
2.Seq2seq,Bilstm,Transformer,Bert,GPT模型的代码以及面试问题收集
3.固态与动态词向量的学习(NNLM,word2vec,Glove,Fasttext)(Elmo,bert,gpt)与面试问题收集
4.算法复习
'''
from torchcrf import CRF








import torch
a=torch.randn(3,4)
print(a)
b=torch.randn(3,4)

print(b)
print(a*b)








