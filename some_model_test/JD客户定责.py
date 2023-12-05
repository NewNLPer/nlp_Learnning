# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/3/20 14:33
coding with comment！！！
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
from transformers import BertTokenizer,BertForSequenceClassification
from sklearn.metrics import accuracy_score,f1_score
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device) + '...')
###数据读取###

def read_data(train_data_path,dev_data_path,predict_path):
    '''
    :param train_data_path: 训练集文件路径，格式为csv，形式为[txt+'/t'+label]
    :param dev_data_path: 测试集
    :param predict_path: 预测集
    :return: train,dev,predict三个数据集的dataframe形式，label的种类数
    '''
    train_df = pd.read_csv(train_data_path, sep="/t", header=None,names=["txt", "label"])
    dev_df=pd.read_csv(dev_data_path, sep="/t",header=None,names=["txt", "label"])
    predict_df=pd.read_csv(predict_path, sep="/t", header=None,names=["txt"])
    nums_label=set(train_df['label'])
    return train_df,dev_df,predict_df,nums_label

train_data_path=''
dev_data_path=''
predict_data_path=''

train_data,dev_data,predict_data,label_set=read_data(train_data_path,dev_data_path,predict_data_path)
###构建标签的映射关系
dic_label={}
for item in label_set:
    dic_label[item]=len(dic_label)

###构建数据数据集
class Mydataset(Data.Dataset):
    def __init__(self,data,station):
        self.data=data
        self.len=len(self.data)
        self.station=station
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        if self.station=='train':
            return self.data['txt'][item,:],self.data['label'][item]
        else:
            return self.data['txt'][item,:]

train_dataset=Mydataset(train_data,'train')

dev_dataset=Mydataset(dev_data,'train')

predict_dataset=Mydataset(predict_data,'predict')

###构建数据迭代器
def collate_fn_train_dev(batch):
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
    tokens_tyep_ids=data['token_type_ids'] # 当前词属于那个句子
    for i in range(len(label)):
        label[i]=dic_label[label[i]]
    return input_ids, attention_mask, tokens_tyep_ids,label

def collate_fn_predict(batch):
    data = [item[0] for item in batch]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=data,
                                       truncation=True,  # 当句子长度大于max_length时，截断
                                       padding='max_length',  # 一律补0到max_length长度
                                       max_length=100,
                                       return_tensors='pt',  # 返回pytorch类型的tensor
                                       return_length=True)  # 返回length，标识长度
    input_ids = data['input_ids']  # input_ids:编码之后的数字
    attention_mask = data['attention_mask']  # attention_mask:补零的位置是0,其他位置是1
    tokens_tyep_ids=data['token_type_ids'] # 当前词属于那个句子

    return input_ids, attention_mask, tokens_tyep_ids



def get_batch_size_change(train_dataset,dev_dataset,predict_dataset,batch_size):
###训练集数据迭代器
    train_dataloader=Data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn_train_dev)

###测试集数据迭代器
    dev_dataloader=Data.DataLoader(
        dataset=dev_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn_train_dev)

###预测集数据迭代器
    predict_dataloader=Data.DataLoader(
        dataset=predict_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn_predict)
    return train_dataloader,dev_dataloader,predict_dataloader


class Pretrain_bert(nn.Module):
    def __init__(self):
        super(Pretrain_bert).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-cased', num_labels=len(label_set))
        self.ST=nn.Softmax(dim=1)

    def forward(self,input_ids,attention_mask,tokens_type_ids):

        out=self.bert(input_ids,attention_mask,tokens_type_ids)

        out=self.ST(out)
        return out

mybert_model=Pretrain_bert()
mybert_model=mybert_model.to(device)
# print(mybert_model)


def train_dev_predict(train_dataset,dev_dataset,predict_dataset,epoch_times,batch_size,optimize,model,loss_fn):
    '''
    :param train_dataloader:
    :param dev_dataloader:
    :param predict_dataloader:
    :param epoch:
    :param batch_size:
    :param optimize:
    :param model:
    :return:
    '''
    train_data,dev_data,predict_data=get_batch_size_change(train_dataset,dev_dataset,predict_dataset,batch_size)
    for epoch in range(1,epoch_times+1):
        sum_loss=0
        iteration_nums=0
        for input_ids,attention_mask,tokens_type_ids,label in train_data:
            out=model(input_ids,attention_mask,tokens_type_ids)
            train_loss=loss_fn(out,label)
            optimize.zero_grad()
            train_loss.backward()
            optimize.step()
            sum_loss+=train_loss.item()
            iteration_nums+=1
        print('第 % epoch time  train_loss :%f'%(epoch,sum_loss/iteration_nums))
        with torch.no_grad():
            pre_label_list=[]
            truth_label_list=[]
            for input_ids,attention_mask,tokens_type_ids,label in dev_data:
                output=model(input_ids,attention_mask,tokens_type_ids)
                _, pre_lab = torch.max(output, 1)
                pre_label_list+=list(pre_lab)
                truth_label_list+=list(label)
            acc=accuracy_score(truth_label_list,pre_label_list)
            f1=f1_score(truth_label_list,pre_label_list)
            print(' epoch_time : %d dev_dataset acc : %f dev_dataset F1 : %f'%(epoch,acc,f1))
    print('--------训练结束，开始对predict_dataset进行预测---------')
    with torch.no_grad():
        result = pd.read_csv(predict_data_path, sep="/t", header=None, names=["txt", "predict_label"])
        predict_list = []
        label_dic=dict(zip(dic_label.keys(),dic_label.values()))
        for input_ids,attention_mask,tokens_type_ids in predict_data:
            output = model(input_ids, attention_mask, tokens_type_ids)
            _, pre_lab = torch.max(output, 1)
            predict_list+=list(pre_lab)
        for i in range(len(predict_list)):
            result['predict_label'][i]=label_dic[predict_list[i]]
            print('-----正在写入%s,已完成%d/%d------'%(predict_data_path,i,len(predict_list)))
        print('----------预测完成，可查看文件-----------')
    return model

loss_func=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mybert_model.parameters(), lr=0.001)
epoch_times=100
batch_size=128

if __name__=='__main__':
    model=train_dev_predict(train_dataset,dev_dataset,predict_dataset,epoch_times,batch_size,
                            optimizer,mybert_model,loss_func)
    # torch.save(model) 保存模型

    # torch.load(model) 加载模型
