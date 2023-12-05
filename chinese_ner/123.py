"""
@author: some_model_test
@time: 2023/3/10 13:57
coding with comment！！！
"""
import random

from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

# print(model)
# token=BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
# model=BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
#
#
#
# print(model)
# exit()
# import pandas as pd
# train_df = pd.read_csv(r"C:\Users\some_model_test\Desktop\train.txt",sep="\t",header=None,names = ["char","label"])#(2169879, 2)训练集
#
# train_df[['char','label']].to_csv(r"C:\Users\some_model_test\Desktop\train.txt",index=False,header=None,sep=' ')

#
# import argparse
#
# parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation test')  # description参数简要描述这个程度做什么以及怎么做。
#
# parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
# parser.add_argument('--seed', type=int, default=72, help='Random seed.')
# parser.add_argument('--abc', type=str, default='a=this is abc', help='Number of epochs to train.')
#
# args = parser.parse_args()
# print(args.sparse)
# print(args.seed)
# print(args.abc)


'''
name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。
action - 命令行遇到参数时的动作，默认值是 store。
store_const，表示赋值为const；
append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
append_const，将参数规范中定义的一个值保存到一个列表；
count，存储遇到的次数；此外，也可以继承 argparse.Action 自定义参数解析；
nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于 Positional argument 使用 default，对于 Optional argument 使用 const；或者是 * 号，表示 0 或多个参数；或者是 + 号表示 1 或多个参数。
const - action 和 nargs 所需要的常量值。
default - 不指定参数时的默认值。
type - 命令行参数应该被转换成的类型。
choices - 参数可允许的值的一个容器。
required - 可选参数是否可以省略 (仅针对可选参数)。
help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
'''


# import torch
# from torchtext import vocab
# word_vectors = vocab.Vectors(r"C:\Users\some_model_test\Desktop\za\gihub\LatticeLSTM-master\data\ctb.50d.vec")###预训练词向量加载
# s='我喜欢吃苹果'
# lanel=[]
# for item in s:
#     lanel.append(word_vectors.stoi[item])
# tenso=torch.tensor(word_vectors.vectors[lanel])
# print(tenso.size())

# emb=nn.Embedding(6,2)
# print(emb.weight)
# a=[[1,2],[3,5]]
# a=torch.tensor(a)
# b=emb(a)
# print(b)


# with open("msra.json", "r", encoding="utf-8") as fh:
#     for i, line in enumerate(fh):
#         if i > 5:
#             continue
#         sample = json.loads(line.strip())
#         print(sample)


# import torch
# from torchtext import vocab
# word_vectors = vocab.Vectors(r"C:\Users\some_model_test\Desktop\za\gihub\Flat-Lattice-Transformer-master\embedding\ctb.50d.vec")
# print(word_vectors.stoi)




# res=[]
# with open(r"C:\Users\some_model_test\Desktop\Chinese-NLP-Corpus-master\NER\Weibo\weibo_test.txt",encoding='utf-8') as f:    #设置文件对象
#     lines = f.readlines()
#     for line in lines:
#         line=line.split('\t')
#         if len(line)==2:
#             res1=line[1].replace('\n','')
#             res.append(line[0][0]+'\t'+res1)
#         else:
#             res.append('\n')
# f=open(r"C:\Users\some_model_test\Desktop\Chinese-NLP-Corpus-master\NER\Weibo\test.txt",'w',encoding='utf-8')
# for line in res:
#     f.write(line+'\n')
# f.close()



list=[1,1,1,1,0,0,0,0]

'''
list=[1,1,1,1,0,0,0,0]如果看作是二分类问题
 0    1
[p1,1-p1] -log(1-p1)
[p2,1-p2] 1-p2
[p3,1-p3] 1-p3
[p4,1-p4] 1-p4
[p5,1-p5] p5
[p6,1-p6] p6
[p7,1-p7] p7




]

'''

import torch.nn as nn
















