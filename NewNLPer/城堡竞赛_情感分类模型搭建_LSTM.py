# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/2/8 19:23
coding with comment！！！
"""
import torch
from torch import nn
import torchtext
from torchtext import vocab
from torchtext.data import Field
import pandas as pd
from sklearn.metrics import f1_score
######################### GPU加速 #########################
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device)+'...')

###在已经完成分词的情况下，文本与标签进行实例化处理。
mytokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=mytokenize,
                  include_lengths=True, use_vocab=True,
                  batch_first=True, fix_length=400)
LABEL = Field(sequential=False, use_vocab=False,
                   pad_token=None, unk_token=None)

text_data_fields = [
("txt", TEXT), # 对文本的操作
("Label", LABEL) # 对标签的操作
]
## 读取数据,将其都转换成实例，维度变化，补全长度
traindata,testdata = torchtext.data.TabularDataset.splits(
    path='C:/Users/NewNLPer/Desktop/za/城堡竞赛', format="csv",
    train="pre_traindata.csv", test="pre_testdata.csv",
    fields=text_data_fields, skip_header=True
)
word_vectors = vocab.Vectors(r"D:\charm\day\NewNLPer\.vector_cache\glove.6B.300d.txt")###预训练词向量加载

#基础查询:word_vectors.vectors(词向量矩阵),word_vectors.stio(词的索引)
TEXT.build_vocab(traindata,vectors=word_vectors)
vocab1=TEXT.vocab #显示所建立的词典
#基础查询:vocab1.vectors(词向量矩阵),vocab1.stio(词的索引)

s=0
for item in vocab1.stoi.keys():
    if item in word_vectors.stoi.keys():
        s+=1
print()
print('训练集词汇表中有%f词语在预训练词向量词表中'%(s/len(vocab1.stoi.keys())))
print()

print('-------------------------原始数据填充中-----------------------------')
result=pd.read_csv(r"C:/Users/NewNLPer/Desktop/za/城堡竞赛/submit_example.csv")
for i in range(25000):
    result['Label'][i]='*'
result[['ID','Label']].to_csv("C:/Users/NewNLPer/Desktop/za/城堡竞赛/submit_example.csv",index=False)
print()
print('-------------------------数据填充完毕-----------------------------')

BATCH_SIZE = 50
train_iter = torchtext.data.BucketIterator(traindata,batch_size = BATCH_SIZE,device=device)
test_iter=torchtext.data.BucketIterator(testdata,batch_size = BATCH_SIZE,device=device,shuffle=False)

# for step, batch in enumerate(test_iter):
#     c1 = batch.txt[0][1]
#     if step == 0:
#         break
# ## 针对一个batch 的数据，可以使用batch.labelcode获得数据的类别标签
# print("数据的类别标签:\n",batch.Label)
# ## batch.cutword[0]是文本对应的标签向量
# print("数据的尺寸:",batch.txt[0].shape)
# ## batch.cutword[1] 对应每个batch使用的原始数据中的索引
# print("数据样本数:",batch.txt[0][1])

class LSTMNet(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim ## lstm神经元个数
        self.layer_dim = layer_dim ## lstm的层数
        ## 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(vocab1.vectors)
        self.embedding.weight.requires_grad = True
        # LSTM ＋ 全连接层
        self.RNN = nn.GRU(embedding_dim, hidden_dim, layer_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(2*hidden_dim,50)
        self.fc1 = nn.Linear(50, output_dim)
    def forward(self, x):
        #x(batch_size,seq_len)--->(batch_size,seq_len,embedding_dim)--->)
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out,_ = self.RNN(embeds, None)   # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        out = self.fc(r_out[:, -1, :])
        out=nn.LayerNorm([out.size()[-2],out.size()[-1]]).cuda()(out)
        out=self.fc1(out.to(device))
        return out
vocab_size = vocab1.vectors.size(0)###词典的长度
embedding_dim = vocab1.vectors.size(1)###嵌入的维度

hidden_dim = 100
layer_dim = 3
output_dim = 2
lstmmodel = LSTMNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)
optimizer = torch.optim.Adam(lstmmodel.parameters(), lr=0.0008)
loss_func = nn.CrossEntropyLoss()   # 损失函数
lstmmodel=lstmmodel.to(device)
print(lstmmodel)

def train_model1(model, traindataloader,criterion,
                 optimizer, num_epochs):
    """
    model:网络模型；
    traindataloader:训练数据集;
    criterion：损失函数；
    optimizer：优化方法；
    num_epochs:训练的轮数
    """
    train_loss_all = []
    train_acc_all = []
    train_f1=[]
    for epoch in range(1,num_epochs+1):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        train_f11=0
        train_num = 0
        model.train()  ## 设置模型为训练模式
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.txt[0], batch.Label.view(-1)
            out = model(textdata.to(device))
            pre_lab = torch.argmax(out, 1)  # 预测的标签
            loss = criterion(out, target.to(device))  # 计算损失函数值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data.to(device))
            train_f11+=f1_score(target.cpu().numpy(),pre_lab.cpu().numpy(),average='macro')
            train_num += len(target)
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        train_f1.append(train_f11/step)
        print('{} Train Loss: {:.6f}  Train Acc: {:.6f} Train f1: {:.6f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1],train_f1[-1]))
train_model1(lstmmodel,train_iter,loss_func,optimizer,num_epochs=1)

result=pd.read_csv(r"C:/Users/NewNLPer/Desktop/za/城堡竞赛/submit_example.csv")
for step, batch in enumerate(test_iter):
    textdata, target = batch.txt[0], batch.Label.view(-1)
    out = lstmmodel(textdata.to(device))
    pre_lab = torch.argmax(out, 1)  # 预测的标签
    for i in range(0,50):
        result['Label'][(step*50)+i]=int(pre_lab[i])
        print('预测数据正在写入submit_example，已完成%d'%((step*50)+i))
print()
result[['ID','Label']].to_csv('C:/Users/NewNLPer/Desktop/za/城堡竞赛/submit_example.csv',index=False)
print('----------预测数据已经全部写入，已保存，请查看！！！----------')


