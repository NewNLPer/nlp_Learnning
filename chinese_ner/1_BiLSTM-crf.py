# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/3/1 13:17
coding with comment！！！
"""
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
from torchcrf import CRF
from sklearn.metrics import f1_score,accuracy_score
import torch.nn.functional as F
import random
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device)+'...')

train_df = pd.read_csv(r"C:\Users\NewNLPer\Desktop\za\城堡竞赛\NER数据\1_train.txt",sep="\t",header=None,names = ["char","label"])#(2169879, 2)训练集



# test_df = pd.read_csv(r"C:\Users\NewNLPer\Desktop\za\城堡竞赛\NER数据\1_test.txt",sep="\t",header=None,names = ["char","label"])#(41560, 2)测试集

#语料按照给定过的句子长度进行划分整合

def pre_process(train_df,sre_len):
    start_zhen=0
    txt=[]
    label=[]
    guall_len=0
    while start_zhen+sre_len<train_df.shape[0]:
        res1=list(train_df['char'][start_zhen:start_zhen+sre_len])
        res2=list(train_df['label'][start_zhen:start_zhen+sre_len])
        end_zhen=start_zhen+sre_len
        while end_zhen<train_df.shape[0]:
            if train_df['label'][end_zhen][0]=='I':
                res1.append(train_df['char'][end_zhen])
                res2.append(train_df['label'][end_zhen])
                end_zhen+=1
            else:
                break
    # while start_zhen+sre_len<train_df.shape[0]:
    #     res1=list(train_df['char'][start_zhen:start_zhen+sre_len])
    #     res2=list(train_df['label'][start_zhen:start_zhen+sre_len])
    #     end_zhen=start_zhen+sre_len
    #     while end_zhen<train_df.shape[0]:
    #         if train_df['label'][end_zhen][0]=='M':
    #             res1.append(train_df['char'][end_zhen])
    #             res2.append(train_df['label'][end_zhen])
    #             end_zhen+=1
    #         elif train_df['label'][end_zhen][0]=='E':
    #             res1.append(train_df['char'][end_zhen])
    #             res2.append(train_df['label'][end_zhen])
    #             end_zhen+=1
    #             break
    #         else:
    #             break
        start_zhen=end_zhen
        txt.append(res1)
        label.append(res2)
        guall_len=max(guall_len,len(res1))
    txt.append(list(train_df['char'][start_zhen:]))
    label.append(list(train_df['label'][start_zhen:]))
    return txt,label,guall_len






# txt=sorted(txt,key=lambda x:len(x),reverse=True)

txt,label,max_len=pre_process(train_df,50)


#构建训练集的词表与标签表
def create_charm_label_vocab(train_df):
    char_dic1={}
    char_dic2={'pad':0,'unk':1}
    label_dic={}#label{标签:编号}
    for i in range(train_df.shape[0]):
        char_dic1[train_df['char'][i]]=char_dic1.get(train_df['char'][i],0)+1
        if train_df['label'][i] not in label_dic:
            label_dic[train_df['label'][i]] =len(label_dic)
    dict_key_ls = list(char_dic1.keys())
    random.shuffle(dict_key_ls)
    for key in dict_key_ls:
        if char_dic1[key]>=1:
            char_dic2[key]=len(char_dic2)
    return char_dic2,label_dic

char_dic,label_dic=create_charm_label_vocab(train_df)


#根据训练集构建的词表来完成txt与label的映射(先不进行填充，mini_batch)

def function_vocab_createz_mask(txt,label,char_dic,label_dic):
    for i in range(len(txt)):
        for j in range(len(txt[i])):
            txt[i][j]=char_dic.get(txt[i][j],1)
            label[i][j]=label_dic[label[i][j]]
    return txt,label

txt,label=function_vocab_createz_mask(txt,label,char_dic,label_dic)







# test_txt,test_label=function_vocab_createz_mask(test_txt,test_label,char_dic,label_dic)

class Mydata(Data.Dataset):
    def __init__(self,data1,data2):
        self.data1=data1
        self.data2=data2
        self.len=len(self.data1)
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        return self.data1[item],self.data2[item]

train_data=Mydata(txt,label)

# test_data=Mydata(test_txt,test_label)


def my_collate(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    res=0
    for item in data:
        res=max(res,len(item))
    mask=[0]*len(data)
    for i in range(len(data)):
        mask[i]=[True]*len(data[i])+[False]*(res-len(data[i]))
        data[i]=data[i]+[0]*(res-len(data[i]))
        label[i]=label[i]+[0]*(res-len(label[i]))
    return torch.tensor(data),label,torch.tensor(mask)


train_loader=Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=128,
    collate_fn=my_collate
)

# for a,b,c in train_loader:
#     print(a.size())
#     print(torch.tensor(b).size())
#     print(c.size())
#     print('==================')
# exit()

# test_loader=Data.DataLoader(
#     dataset=test_data,
#     batch_size=128,
#     shuffle=True,
#     collate_fn=my_collate
# )

class MyBiGRU_crf(nn.Module):
    def __init__(self,embeding_dim,hidden_dim,tags_nums,vocab_size):
        super(MyBiGRU_crf,self).__init__()
        self.embeding=embeding_dim
        self.hidden_dim=hidden_dim
        self.tags_nums=tags_nums
        self.vocab_size=vocab_size
        self.EMD=nn.Embedding(self.vocab_size,self.embeding,padding_idx=0)
        self.RNN=nn.LSTM(self.embeding,self.hidden_dim,bidirectional=True,batch_first=True)
        self.MLP1=nn.Linear(2*self.hidden_dim,self.tags_nums)
        # self.MLP2=nn.Linear(100,self.tags_nums)
        self.crf = CRF(self.tags_nums)
        # self.ma=nn.Softmax(dim=2)

    def get_GRU(self,data):
        emd=self.EMD(data)
        h_n,_=self.RNN(emd)###正反向拼接
        out=self.MLP1(h_n.to(device))
        # out=self.ma(out)
        return out

    def loss_f(self,data,tags,mask):
        emission=self.get_GRU(data)
        print(emission.size())
        print(tags.size())
        print(mask.size())
        exit()
        loss_value=sum(self.crf.forward(emission,tags,mask))/data.size()[0]
        return loss_value

    def forward(self,data,mask):
        return self.crf.viterbi_decode(self.get_GRU(data),mask)

embeding_dim=100
hidden_dim=256
tags_nums=len(label_dic)
vocab_size=len(char_dic)
model=MyBiGRU_crf(embeding_dim,hidden_dim,tags_nums,vocab_size)
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.02)
epoch_time=10

print(model)


def train(model,optimizer,epoch_time,train_loader):
    for epoch in range(1,epoch_time+1):
        train_loss=0
        test_f1=0
        test_acc=0
        p=0
        for txt,label,mask in train_loader:
            batch_loss = -model.loss_f(txt.to(device), torch.tensor(label).to(device), mask.to(device))
            train_loss+=batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            p+=1
        print(train_loss/p)
        # with torch.no_grad():
        #     p1=0
        #     for txt,label,mask in test_loader:
        #         label1=[]
        #         pre1=[]
        #         pre=model(txt.to(device),mask.to(device))
        #         for i in range(len(pre)):
        #             pre1+=pre[i]
        #             label1+=label[i][:len(pre[i])]
        #         test_f1 += f1_score(label1,pre1,average='macro')
        #         test_acc += accuracy_score(label1,pre1)
        #         p1+=1
        #     print('=====================================================================================================')
        #     print('  Train epoch: %d time , Train loss: %f , Test_F1_score: %f , Test_acc_score : %f ' % (epoch, train_loss / p,test_f1/p1,test_acc/p1))

train(model,optimizer,epoch_time,train_loader)

if __name__ == '__main__':
    s='费丹阳虽然去了大连理工大学，但是她最喜欢的还是上海财经大学,虽然她现在目前在北京，喜欢的还是上海。'
    print(s)
    mask=[True]*len(s)
    mask+=[False]*(max_len-len(mask))
    mask=torch.tensor(mask).view(1,-1)
    res=[]
    label_num = dict(zip(label_dic.values(), label_dic.keys()))
    for item in s:
        res.append(char_dic.get(item,1))
    res=torch.tensor(res).view(1,-1)
    with torch.no_grad():
        pre=model(res.to(device),mask.to(device))
        print(pre)
        for i in range(len(pre[0])):
            print(s[i],label_num[pre[0][i]])



