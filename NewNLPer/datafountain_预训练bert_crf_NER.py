# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/4/10 9:51
coding with comment！！！
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import  BertForPreTraining, BertTokenizer
from torchcrf import CRF
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device) + '...')

label_dic = {'B-GPE': 0, 'M-GPE': 1, 'E-GPE': 2, 'O': 3, 'B-PER': 4, 'M-PER': 5, 'E-PER': 6, 'B-LOC': 7, 'M-LOC': 8,
             'E-LOC': 9, 'B-ORG': 10, 'M-ORG': 11, 'E-ORG': 12, 'S-GPE': 13, 'S-LOC': 14, 'S-PER': 15, 'S-ORG': 16}

model=BertForPreTraining.from_pretrained('bert-base-chinese')

train_data=pd.read_csv(r"C:\Users\NewNLPer\Desktop\char_ner_train.csv",encoding='utf-8',keep_default_na=False)
train_data.columns=['txt','tag']
predict_data=pd.read_csv(r"C:\Users\NewNLPer\Desktop\evaluation_public.csv",encoding='utf-8',keep_default_na=False)
predict_data.columns=['id','txt']

# txt=[]
# label=[]
# for i in range(len(predict_data)):
#     if predict_data['txt'][i]:
#         txt.append(str(predict_data['txt'][i]))
#         label.append(str(predict_data['id'][i]))
#
# with open(r"C:\Users\NewNLPer\Desktop\2.txt",'w') as f:
#     for i in range(len(txt)):
#         if txt[i]==',' or not txt[i]:
#             f.write('*'+','+label[i]+'\n')
#         else:
#             f.write(txt[i] + ',' + label[i] + '\n')
#             print('%d/%d'%(i+1,len(txt)))


train_data=pd.read_table(r"C:\Users\NewNLPer\Desktop\1.txt",encoding='gbk',names=['txt','tag'],sep=',')
predict_data=pd.read_table(r"C:\Users\NewNLPer\Desktop\2.txt",encoding='gbk',names=['id','txt'],sep=',')

print(train_data)
print(predict_data)

exit()

def get_data_list(path):
    data_frame = pd.read_csv(path, keep_default_na=False)  ##'character', 'tag'\
    data_frame=data_frame.dropna()
    a,b=data_frame.columns
    if a!='character': #始终保证 a==character
        b, a = data_frame.columns

    char_list=[]
    label_list=[]

    char_stack=[]
    label_stack=[]

    for i in range(len(data_frame)):
        if data_frame[a][i]:
            char_stack.append(data_frame[a][i])
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

train_path=r"C:\Users\NewNLPer\Desktop\train_ner_data.txt"
predict_path=r"C:\Users\NewNLPer\Desktop\predict_ner_data.txt"


train_data,label_data,=get_data_list(train_path)

predict_data,predict_label=get_data_list(predict_path)

max_len=0
for i in range(len(predict_data)):
    max_len=max(max_len,len(predict_data[i]))
    # if max_len==527:
    #     print(predict_data[i])
    #     print(predict_label[i])
    #     print(int(predict_label[i][0])+max_len//2)
    #     break
print(max_len)
exit()



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








def collate_fn(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    res=0
    for i in range(len(data)):
        data[i]=''.join(data[i])
        res=max(res,len(data[i]))
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=data,
                                       truncation=True,  # 当句子长度大于max_length时，截断
                                       padding='max_length',  # 一律补0到max_length长度
                                       max_length=res+2,
                                       return_tensors='pt',  # 返回pytorch类型的tensor
                                       return_length=True)  # 返回length，标识长度
    input_ids = data['input_ids']  # input_ids:编码之后的数字
    attention_mask = data['attention_mask']  # attention_mask:补零的位置是0,其他位置是1
    token_type_ids=data['token_type_ids']
    for i in range(len(label)):
        label[i] = [3] + label[i] + [3] + [3] * (res- len(label[i]))
    return input_ids, attention_mask,token_type_ids, label

train_loader = Data.DataLoader(
    dataset=train_data_set,
    shuffle=False,
    batch_size=8,
    collate_fn=collate_fn)


predict_loader=Data.DataLoader(
    dataset=predict_data_set,
    shuffle=False,
    batch_size=8,
    collate_fn=collate_fn
)



res=0
for a,b,c,d in predict_loader:
    print(a.size())
    print(b.size())
    print(c.size())
    res=max(res,c.size()[1])
    print('===================')
print(res)
exit()
'''

torch.Size([8, 532])
torch.Size([8, 532])
torch.Size([8, 532])

'''



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
        label=torch.tensor(label)
        return -sum(self.crf.forward(out,label,atten_mask))/input_ids.size()[0]

    def forward(self,input_ids,atten_mask,token_ids):
        out = self.get_bert(input_ids, atten_mask, token_ids)
        return self.crf.viterbi_decode(out,atten_mask)

model=Bert_CRF()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.000002)
print(model)
epoch_nums=1


def train(epoch_nums,train_loader,model,optimizer):
    for epoch in range(1,epoch_nums+1):
        train_loss=0
        p=0
        for input_ids,attention_maks,token_type_ids,label in train_loader:
            input_ids=input_ids.to(device)
            attention_maks=attention_maks.to(device)
            token_type_ids=token_type_ids.to(device)
            # label=label.to(device)
            loss_f=model.get_loss(input_ids,attention_maks,token_type_ids,label)
            train_loss += loss_f.item()
            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()
            p+=1
            if p%100==0:
                print('%d/2000'%(p))
        print('Train epoch : %d | Train loss : %f'%(epoch,train_loss/p))
    print('==================================开始预测写入=======================================')
    try:
        with torch.no_grad():
            result = pd.read_csv(r"C:\Users\NewNLPer\Desktop\ner\submit_example.csv")
            ans = []
            res = 0
            my_dic = dict(zip(label_dic.values(), label_dic.keys()))
            for input_ids, attention_maks, token_type_ids,_ in predict_loader:
                input_ids=input_ids.to(device)
                attention_maks=attention_maks.to(device)
                token_type_ids=token_type_ids.to(device)
                pre = model(input_ids,attention_maks,token_type_ids)
    except:
        print(input_ids.size())
        print(attention_maks.size())
        print(token_type_ids.size())
        exit()
        #     # print(pre)
        #     # exit()
        #     for i in range(len(pre)):
        #         pre[i] = pre[i] + [' ']
        #         res += len(pre[i])
        #         ans += pre[i]
        #
        # for i in range(len(ans)):
        #     if ans[i] == ' ':
        #         continue
        #     result['tag'][i] = my_dic[ans[i]]
        #
        #     print('正在写入文件，%d/%d' % (i + 1, len(ans)))
        # result[['id', 'tag']].to_csv(r"C:\Users\NewNLPer\Desktop\ner\123.csv", index=False)
        # print('已写入，请查看%s' % (r"C:\Users\NewNLPer\Desktop\ner\123.csv"))

if __name__=='__main__':
    train(epoch_nums,train_loader,model,optimizer)

