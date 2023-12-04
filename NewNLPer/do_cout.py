# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/4/6 13:31
coding with comment！！！
"""
import torch
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
from torchcrf import CRF
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device)+'...')

label_dic = {'B-GPE': 0, 'M-GPE': 1, 'E-GPE': 2, 'O': 3, 'B-PER': 4, 'M-PER': 5, 'E-PER': 6, 'B-LOC': 7, 'M-LOC': 8,
             'E-LOC': 9, 'B-ORG': 10, 'M-ORG': 11, 'E-ORG': 12, 'S-GPE': 13, 'S-LOC': 14, 'S-PER': 15, 'S-ORG': 16}



def get_char_dic(path=r"C:\Users\NewNLPer\Desktop\train.csv"):
    train_data = pd.read_csv(path, keep_default_na=False)  ##'character', 'tag'\
    dic_c = {'pad': 0, 'unk': 1}
    for i in range(len(train_data)):
        if train_data['character'][i] not in dic_c:
            dic_c[train_data['character'][i]] = len(dic_c)
    return dic_c

char_dic=get_char_dic()



def get_data_list(path):
    data_frame = pd.read_csv(path, keep_default_na=False)  ##'character', 'tag'\
    a,b=data_frame.columns
    if a!='character': #始终保证 a==character
        b, a = data_frame.columns

    char_list=[]
    label_list=[]

    char_stack=[]
    label_stack=[]

    for i in range(len(data_frame)):
        if data_frame[a][i]:
            char_stack.append(char_dic.get(data_frame[a][i],1))
            if b=='id':
                label_stack.append(data_frame[b][i])
            else:
                label_stack.append(label_dic[data_frame[b][i]])
        else:
            char_list.append(char_stack)
            label_list.append(label_stack)
            char_stack=[]
            label_stack=[]
    return char_list,label_list

train_path=r"C:\Users\NewNLPer\Desktop\train.csv"

predict_path=r"C:\Users\NewNLPer\Desktop\dev.csv"

train_data,label_data,=get_data_list(train_path)

predict_data,predict_label=get_data_list(predict_path)



class Mydataset(Data.Dataset):

    def __init__(self,char_train_data,label_train_data):
        self.char_train_data=char_train_data
        self.label_train_data = label_train_data

    def __len__(self):
        return len(self.char_train_data)

    def __getitem__(self, index):
        return self.char_train_data[index],self.label_train_data[index]


train_data_set= Mydataset(train_data,label_data)
predict_data_set= Mydataset(predict_data,predict_label)


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
        label[i]=label[i]+[3]*(res-len(label[i]))

    return torch.tensor(data),label,torch.tensor(mask)

train_loader=Data.DataLoader(
    dataset=train_data_set,
    shuffle=False,
    batch_size=32,
    collate_fn=my_collate
)


predict_loader=Data.DataLoader(
    dataset=predict_data_set,
    shuffle=False,
    batch_size=32,
    collate_fn=my_collate
)


class MyBiGRU_crf(nn.Module):
    def __init__(self,embeding_dim,hidden_dim,tags_nums,vocab_size):
        super(MyBiGRU_crf,self).__init__()
        self.embeding=embeding_dim
        self.hidden_dim=hidden_dim
        self.tags_nums=tags_nums
        self.vocab_size=vocab_size
        self.EMD=nn.Embedding(self.vocab_size,self.embeding)
        self.RNN=nn.LSTM(self.embeding,self.hidden_dim,bidirectional=True,batch_first=True)
        self.MLP1=nn.Linear(2*self.hidden_dim,self.tags_nums)
        # self.MLP2=nn.Linear(100,self.tags_nums)
        self.crf = CRF(self.tags_nums)
        # self.ma=nn.Softmax(dim=2)

    def get_GRU(self,data):
        emd=self.EMD(data)
        h_n,_=self.RNN(emd)
        out=self.MLP1(h_n.to(device))
        # out=self.ma(out)
        return out

    def loss_f(self,data,tags,mask):
        emission=self.get_GRU(data)
        loss_value=sum(self.crf.forward(emission,tags,mask))/data.size()[0]
        return loss_value

    def forward(self,data,mask):
        return self.crf.viterbi_decode(self.get_GRU(data),mask)

embeding_dim=100
hidden_dim=64
tags_nums=len(label_dic)
vocab_size=len(char_dic)

model=MyBiGRU_crf(embeding_dim,hidden_dim,tags_nums,vocab_size)
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)
epoch_time=50

print(model)

def train(model,optimizer,epoch_time,train_loader):
    for epoch in range(1,epoch_time+1):
        train_loss=0
        p=0
        for txt,label,mask in train_loader:
            batch_loss = -model.loss_f(txt.to(device), torch.tensor(label).to(device), mask.to(device))
            train_loss+=batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            p+=1
        print('epoch_time : %d train loss : %f'%(epoch,train_loss/p))
    print('==================================开始预测写入=======================================')
    with torch.no_grad():
        result = pd.read_csv(r"C:\Users\NewNLPer\Desktop\submit_example.csv")
        ans=[]
        res=0
        my_dic = dict(zip(label_dic.values(), label_dic.keys()))
        for txt,label,mask in predict_loader:
            pre=model(txt.to(device),mask.to(device))
            for i in range(len(pre)):
                pre[i]=pre[i]+[' ']
                res+=len(pre[i])
                ans+=pre[i]


        for i in range(len(ans)):
            if ans[i]==' ':
                continue
            result['tag'][i]=my_dic[ans[i]]
            print('正在写入文件，%d/%d'%(i+1,len(ans)))
        result[['id', 'tag']].to_csv(r"C:\Users\NewNLPer\Desktop\123.csv",index=False)
        print('已写入，请查看%s'%('C:/Users/NewNLPer/Desktop/submit_example - 副本.csv'))

if __name__ == '__main__':
    train(model,optimizer,epoch_time,train_loader)













