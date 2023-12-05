# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/4/10 9:46
coding with comment！！！
"""
import torch
import pandas as pd
import torch.utils.data as Data
from sklearn.metrics import accuracy_score,f1_score
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device) + '...')

model= RobertaModel.from_pretrained('roberta-base')



def read_data(train_data_path,dev_data_path,predict_path):
    '''
    :param train_data_path: 训练集文件路径，格式为csv，形式为[txt+'/t'+label]
    :param dev_data_path: 测试集
    :param predict_path: 预测集
    :return: train,dev,predict三个数据集的dataframe形式，label的种类数
    '''
    train_data = pd.read_csv(train_data_path,encoding='gb18030')
    train_data.columns = ['txt', 'label']
    dev_data = pd.read_csv(dev_data_path, encoding='gb18030')
    dev_data.columns = ['txt', 'label']
    predict_data = pd.read_csv(predict_path,encoding='utf-8')
    predict_data.columns = ['txt', 'label']
    return train_data,dev_data,predict_data


train_data_path=r"C:\Users\NewNLPer\Desktop\竞赛\Classcify\pre_traindata.csv"
dev_data_path=r"C:\Users\NewNLPer\Desktop\竞赛\Classcify\pre_devdata.csv"
predict_data_path=r"C:\Users\NewNLPer\Desktop\竞赛\Classcify\pre_testdata.csv"

train_data,dev_data,predict_data=read_data(train_data_path,dev_data_path,predict_data_path)


class Mydataset(Data.Dataset):
    def __init__(self,data):
        self.data=data
        self.len=len(self.data)
    def __len__(self):
        return self.len
    def __getitem__(self, item):
            return self.data['txt'][item],self.data['label'][item]


train_dataset=Mydataset(train_data)
dev_dataset=Mydataset(dev_data)
predict_dataset=Mydataset(predict_data)

def collate_fn_train_dev(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=data,
                                       truncation=True,  # 当句子长度大于max_length时，截断
                                       padding='max_length',  # 一律补0到max_length长度
                                       max_length=128,
                                       return_tensors='pt',  # 返回pytorch类型的tensor
                                       return_length=True)  # 返回length，标识长度
    input_ids = data['input_ids']  # input_ids:编码之后的数字
    attention_mask = data['attention_mask']  # attention_mask:补零的位置是0,其他位置是1
    return input_ids, attention_mask,label

batch_size=1

train_dataloader=Data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn_train_dev)

dev_dataloader=Data.DataLoader(
        dataset=dev_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn_train_dev)

predict_dataloader=Data.DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_train_dev)###文本跟标签不应该被打乱


class Pretrain_bert(nn.Module):
    def __init__(self):
        super(Pretrain_bert,self).__init__()
        self.bert =  RobertaModel.from_pretrained('roberta-base')
        self.mlp=nn.Linear(768,2)

    def forward(self,input_ids,attention_mask):
        out=self.bert(input_ids,attention_mask)
        out=out.pooler_output
        return self.mlp(out)

mybert_model=Pretrain_bert()

mybert_model=mybert_model.to(device)

loss_func=nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(mybert_model.parameters(), lr=0.0001)

epoch_times=5

def train_dev_predict(train_dataloader,dev_dataloader,predict_dataloader,epoch_times,optimize,model,loss_fn,save_path):
    '''
    :param train_dataloader:
    :param dev_dataloader:
    :param predict_dataloader:
    :param epoch:
    :param optimize:
    :param model:
    :return:
    '''
    for epoch in range(1,epoch_times+1):
        sum_loss=0
        iteration_nums=0
        for input_ids,attention_mask,label in train_dataloader:
            label=torch.tensor(label).to(device)
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            out=model(input_ids,attention_mask)
            train_loss=loss_fn(out,label)
            optimize.zero_grad()
            train_loss.backward()
            optimize.step()
            sum_loss+=train_loss.item()
            iteration_nums+=1
            if iteration_nums%1000==0:
                print('epoch_time : %d  当前epoch进度 : %d/19998'%(epoch,iteration_nums))
        print(' --- train-process  epcoh_time : %d  train_loss :%f'%(epoch,sum_loss/iteration_nums))
        with torch.no_grad():
            pre_label_list=[]
            truth_label_list=[]
            for input_ids,attention_mask,label in dev_dataloader:
                input_ids=input_ids.to(device)
                attention_mask=attention_mask.to(device)
                output=model(input_ids,attention_mask)
                _, pre_lab = torch.max(output, 1)

                pre_lab=pre_lab.tolist()
                pre_label_list+=list(pre_lab)
                truth_label_list+=list(label)

            acc=accuracy_score(truth_label_list,pre_label_list)
            f1=f1_score(truth_label_list,pre_label_list)

            print(' --- dev-process  epoch_time : %d dev_dataset acc : %f dev_dataset F1 : %f'%(epoch,acc,f1))
            print('=======================================================================================================')
    print('--------训练结束，开始对 predict_dataloader 进行预测---------')
    exit()
    with torch.no_grad():
        result = pd.read_csv(save_path)
        predict_list = []
        for input_ids,attention_mask,tokens_type_ids,mark_nums in predict_dataloader:
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            tokens_type_ids=tokens_type_ids.to(device)
            output = model(input_ids, attention_mask, tokens_type_ids)
            _, pre_lab = torch.max(output, 1)
            pre_lab=pre_lab.tolist()
            predict_list+=list(pre_lab)
        for i in range(len(predict_list)):
            result['Label'][i]=predict_list[i]
            print('-----正在写入%s,已完成%d/%d------'%(save_path,i+1,len(predict_list)))
        print('----------预测完成，可查看文件-----------')
        result[['ID', 'Label']].to_csv(save_path, index=False)

if __name__=='__main__':
    save_path=r"C:\Users\NewNLPer\Desktop\竞赛\Classcify\submit_example.csv"
    train_dev_predict(train_dataloader,dev_dataloader,predict_dataloader,epoch_times,
                            optimizer,mybert_model,loss_func,save_path)


