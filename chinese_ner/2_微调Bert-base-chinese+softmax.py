# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/3/7 15:16
coding with comment！！！
"""
import pandas as pd
import torch
import torch.utils.data as Data
from transformers import BertForTokenClassification
import random
from transformers import BertTokenizer, BertTokenizerFast
from torchcrf import CRF
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
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

# res=tokenizer.get_vocab()
# print(res['[CLS]'])-->101
# print(res['[PAD]'])-->0
# print(res['[SEP]'])-->102
# exit()

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
    for i in range(len(label)):
        label[i] = [-100] + label[i] + [-100] + [-100] * (98 - len(label[i]))
    return input_ids, attention_mask, torch.tensor(label)


train_loader = Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=64,
    collate_fn=collate_fn)


# for a,b,c in train_loader:
#     print(a.size())
#     print(b.size())
#     print(c)
#     break
# exit()

class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            'bert-base-cased', num_labels=len(label_dic))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        print(output[1].size())
        exit()
        return output
    #     self.crf=CRF(len(label_dic))
    # def get_bert(self, input_id, mask, label):
    #     _, output = self.bert(input_ids=input_id, attention_mask=mask,
    #                           labels=label, return_dict=False)
    #     return output
    #
    # def loss_f(self, input_id, mask, label):
    #     emssion = self.get_bert(input_id, mask, label)
    #     mask=mask.bool()
    #     loss_value=sum(self.crf.forward(emssion,label,mask))
    #     return -1*loss_value
    #
    # def forward(self,input_id,mask):
    #     return self.crf.viterbi_decode(self.get_bert(input_id,label=None),mask.bool())





model = BertModel()
model = model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

epoch_step = 100


# for a,b,c in train_loader:
#     print(a.size())
#     print(b.size())
#     print(c.size())
#
# exit()

def train_pross(model, epoch_step, optimizer, train_loader):
    for epoch_num in range(1, epoch_step + 1):
        total_loss_train = 0
        total_acc_train = 0
        p = 0
        for input_ids, attention_mask, label in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            loss, logits = model(input_ids, attention_mask, label)
            # print(logits.size())
            # print(attention_mask.size())
            # print(attention_mask)
            # print(attention_mask.bool())
            # exit()
            logits_clean = logits[label != -100]
            predictions = logits_clean.argmax(dim=1)
            label_clean = label[label != -100]
            acc = (predictions == label_clean).float().mean()
            total_loss_train += loss.item()
            total_acc_train += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p += 1
        print('epoch time : %d | train_loss : %f | train_acc : %f' % (
        epoch_num, total_loss_train / p, total_acc_train / p))


train_pross(model, epoch_step, optimizer, train_loader)
