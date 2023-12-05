# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/3/8 17:02
coding with comment！！！
"""
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from transformers import BertTokenizer, BertForPreTraining
import torch
import random
from transformers import BertTokenizer, BertTokenizerFast
from torchcrf import CRF
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model=BertForPreTraining.from_pretrained('bert-base-chinese')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device) + '...')
train_df = pd.read_csv(r"C:\Users\NewNLPer\Desktop\za\城堡竞赛\NER数据\3.txt", sep=" ", header=None,
                       names=["char", "label"])  # (2169879, 2)训练集

def pre_process(train_df, sre_len):
    start_zhen = 0
    txt = []
    label = []
    guall_len = 0
    while start_zhen + sre_len < train_df.shape[0]:
        res1 = list(train_df['char'][start_zhen:start_zhen + sre_len])
        res2 = list(train_df['label'][start_zhen:start_zhen + sre_len])
        end_zhen = start_zhen + sre_len
        while end_zhen < train_df.shape[0]:
            if train_df['label'][end_zhen][0] == 'M':
                res1.append(train_df['char'][end_zhen])
                res2.append(train_df['label'][end_zhen])
                end_zhen += 1
            elif train_df['label'][end_zhen][0] == 'E':
                res1.append(train_df['char'][end_zhen])
                res2.append(train_df['label'][end_zhen])
                end_zhen += 1
                break
            else:
                break
        start_zhen = end_zhen
        txt.append(res1)
        label.append(res2)
        guall_len = max(guall_len, len(res1))
    txt.append(list(train_df['char'][start_zhen:]))
    label.append(list(train_df['label'][start_zhen:]))
    return txt, label, guall_len

txt, label, max_len = pre_process(train_df, 50)  # max_src_len=79


def create_charm_label_vocab(train_df):
    char_dic1 = {}
    char_dic2 = {'pad': 0, 'unk': 1}
    label_dic = {}  # label{标签:编号}
    for i in range(train_df.shape[0]):
        char_dic1[train_df['char'][i]] = char_dic1.get(train_df['char'][i], 0) + 1
        if train_df['label'][i] not in label_dic:
            label_dic[train_df['label'][i]] = len(label_dic)
    dict_key_ls = list(char_dic1.keys())
    random.shuffle(dict_key_ls)
    for key in dict_key_ls:
        if char_dic1[key] >= 1:
            char_dic2[key] = len(char_dic2)
    return char_dic2, label_dic

char_dic, label_dic = create_charm_label_vocab(train_df)

def function_vocab_createz_mask(label, label_dic):
    for i in range(len(txt)):
        for j in range(len(txt[i])):
            label[i][j] = label_dic[label[i][j]]
    return label


label = function_vocab_createz_mask(label, label_dic)
for i in range(len(txt)):
    txt[i] = ''.join(txt[i])


class Mydata(Data.Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.len = len(self.data1)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data1[item], self.data2[item]

train_data = Mydata(txt, label)

def collate_fn(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=data,
                                       truncation=True,  # 当句子长度大于max_length时，截断
                                       padding='max_length',  # 一律补0到max_length长度
                                       max_length=100,
                                       return_tensors='pt',  # 返回pytorch类型的tensor
                                       return_length=True)  # 返回length，标识长度
    input_ids = data['input_ids']  # input_ids:编码之后的数字
    attention_mask = data['attention_mask']  # attention_mask:补零的位置是0,其他位置是1
    token_type_ids=data['token_type_ids']
    for i in range(len(label)):
        label[i] = [0] + label[i] + [0] + [0] * (98 - len(label[i]))
    return input_ids, attention_mask,token_type_ids, torch.tensor(label)

train_loader = Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=64,
    collate_fn=collate_fn)

# out=tokenizer('----',return_tensors='pt')
# res=model(**out).prediction_logits#(batch_size,src_len,100)

class Bert_CRF(nn.Module):
    def __init__(self,):
        super(Bert_CRF, self).__init__()
        self.bert=BertForPreTraining.from_pretrained('bert-base-chinese')
        self.crf=CRF(len(label_dic))
        self.MLP=nn.Linear(21128,len(label_dic))

    def get_bert(self,input_ids,atten_mask,token_ids):
        out=self.bert(input_ids,atten_mask,token_ids).prediction_logits
        return self.MLP(out)

    def get_loss(self,input_ids,atten_mask,token_ids,label):
        out=self.get_bert(input_ids,atten_mask,token_ids)
        atten_mask=atten_mask.bool()
        return -sum(self.crf.forward(out,label,atten_mask))/input_ids.size()[0]


    def forward(self,input_ids,atten_mask,token_ids):
        out = self.get_bert(input_ids, atten_mask, token_ids)
        return self.crf.viterbi_decode(out,atten_mask)

model=Bert_CRF()
model=model.to(device)







optimizer=torch.optim.Adam(model.parameters(),lr=0.000002)
print(model)
epoch_nums=150

def train(epoch_nums,train_loader,model,optimizer):
    for epoch in range(1,epoch_nums+1):
        train_loss=0
        p=0
        for input_ids,attention_maks,token_type_ids,label in train_loader:
            input_ids=input_ids.to(device)
            attention_maks=attention_maks.to(device)
            token_type_ids=token_type_ids.to(device)
            label=label.to(device)
            loss_f=model.get_loss(input_ids,attention_maks,token_type_ids,label)
            train_loss += loss_f.item()
            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()
            p+=1
        print('Train epoch : %d | Train loss : %f'%(epoch,train_loss/p))
train(epoch_nums,train_loader,model,optimizer)
if __name__ == '__main__':
    s='费丹阳虽然考上了大连理工大学，但是她还是想去上海财经大学,虽然她现在目前在北京，喜欢的还是纪杰周。'
    out=tokenizer(s,return_tensors='pt')
    input_ids=out['input_ids'].to(device)
    token_type_ids=out['token_type_ids'].to(device)
    attention_mask=out['attention_mask']
    attention_mask=attention_mask.bool().to(device)
    pre_label=model(input_ids,attention_mask,token_type_ids)[0]
    s='C'+s+'S'
    dic_label = dict(zip(label_dic.values(), label_dic.keys()))
    for i in range(len(pre_label)):
        print(s[i]+'-'+dic_label[pre_label[i]])



