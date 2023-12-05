"""
@author: some_model_test
@time: 2022/12/28 18:32
coding with comment！！！
"""
import torch
from torchcrf import CRF
import torch.nn as nn
import copy

'''
费 B_person
丹 I_person
阳 I_person
最 O
喜 O
欢 O
吃 O
唐 B_location
山 I_location
冒 O
菜 O

纪 B_person
杰 I_person
周 I_person
想 O
去 O
北 B_location
京 I_location
天 O
安 O
门 O
'''
orinal=['费丹阳最喜欢吃唐山冒菜','纪杰周想去北京天安门']
ner=[['B_person','I_person','I_person','O','O','O','O','B_location','I_location','O','O'],
     ['B_person','I_person','I_person','O','O','B_location','I_location','O','O','O']]



dic_vocab1={}#标号
dic_vocab2={}#字频计算
set_vocab=set()#集合存储
for i in range(2):
    for j in range(len(orinal[i])):
        dic_vocab2[orinal[i][j]]=dic_vocab2.get(orinal[i][j],0)+1
        if orinal[i][j] not in set_vocab:
            dic_vocab1[orinal[i][j]]=len(set_vocab)+2
            set_vocab.add(orinal[i][j])
dic_vocab1['pad']=0
dic_vocab1['unk']=1

for i in range(2):
    res=[0]*11
    for j in range(len(orinal[i])):
        res[j]=dic_vocab1[orinal[i][j]]
    orinal[i]=res

dic_label1={}#标号
dic_label2={}#字频计算
set_label=set()#集合存储

for i in range(2):
    for j in range(len(ner[i])):
        dic_label2[ner[i][j]]=dic_label2.get(ner[i][j],0)+1
        if ner[i][j] not in set_label:
            dic_label1[ner[i][j]]=len(set_label)
            set_label.add(ner[i][j])
for i in range(2):
    res=[0]*11
    for j in range(len(ner[i])):
        res[j]=dic_label1[ner[i][j]]
    ner[i]=res


mask=copy.deepcopy(orinal)
for i in range(2):
    for j in range(len(mask[i])):
        if mask[i][j]!=0:
            mask[i][j]=True
        else:
            mask[i][j]=False


orinal=torch.tensor(orinal)
mask=torch.tensor(mask)
ner=torch.tensor(ner)

# print(orinal)
# print('========')
# print(ner)
# print('=========')
# print(mask)



class MyBiGRU_crf(nn.Module):
    def __init__(self,embeding_dim,hidden_dim,tags_nums,vocab_size):
        super(MyBiGRU_crf,self).__init__()
        self.embeding=embeding_dim
        self.hidden_dim=hidden_dim
        self.tags_nums=tags_nums
        self.vocab_size=vocab_size
        self.EMD=nn.Embedding(self.vocab_size,self.embeding)
        self.RNN=nn.GRU(self.embeding,self.hidden_dim,bidirectional=True,batch_first=True)
        self.MLP=nn.Linear(2*self.hidden_dim,tags_nums)
        self.crf=CRF(tags_nums)

    def get_GRU(self,data):
        emd=self.EMD(data)
        h_n,_=self.RNN(emd)
        out=self.MLP(h_n)
        return out
    def loss_f(self,data,tags,mask):
        return -sum(self.crf(self.get_GRU(data),tags,mask))
    def forward(self,data,mask):

        return self.crf.viterbi_decode(self.get_GRU(data),mask)

embeding_dim=20
hidden_dim=50
tags_nums=5
vocab_size=len(dic_vocab1)

model=MyBiGRU_crf(embeding_dim,hidden_dim,tags_nums,vocab_size)
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

# for epoch in range(1,21):
#     pre=model(orinal,mask)
#     loss=model.loss_f(orinal,ner,mask)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print('第%d次迭代，loss为:%f'%(epoch,loss.item()))





ans='费丹阳去过唐山'
ans1=[0]*11
mask1=[False]*11
for i in range(len(ans)):
    if ans[i] in dic_vocab1:
        ans1[i]=dic_vocab1[ans[i]]
        mask1[i]=True
    else:
        ans1[i]=1
        mask1[i]=True



ans1=torch.tensor(ans1)
ans1=ans1.view(1,-1)
mask1=torch.tensor(mask1)
mask1=mask1.view(1,-1)
pre=model(ans1,mask1)

pre=list(pre[0])
ans1=list(ans1[0])

rs = dict(zip(dic_label1.values(), dic_label1.keys()))
for i in range(len(ans)):
    print(ans[i],rs[pre[i]])











# print(model)
#
#
#
#
# print('==================================================')
# nums_tags=5
# model=CRF(nums_tags)
# emssion=torch.randn(2,3,5) #(batch_size,sre_len,nums_tags)
# mask=torch.tensor([[True,True,True],[True,True,True]])
# tags=torch.tensor([[1,2,3],[0,4,3]])
# a=model(emssion,tags,mask)
# print(a)###列表中，每个batch的loss
# print(mask.size())
# print(model.viterbi_decode(emssion,mask))###寻找最优路径