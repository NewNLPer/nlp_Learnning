# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/2/9 21:31
coding with comment！！！
"""
import torch
from torch import nn
import torchtext
from torchtext import vocab
from torchtext.data import Field
import pandas as pd
import torch.optim as optim
import math
import numpy as np
import gc


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
    path=r'C:\Users\NewNLPer\Desktop\za\城堡竞赛\情感分类数据', format="csv",
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
result=pd.read_csv(r"C:\Users\NewNLPer\Desktop\za\城堡竞赛\情感分类数据\submit_example.csv")
for i in range(25000):
    result['Label'][i]='*'
result[['ID','Label']].to_csv(r"C:\Users\NewNLPer\Desktop\za\城堡竞赛\情感分类数据\submit_example.csv",index=False)
print()
print('-------------------------数据填充完毕-----------------------------')


BATCH_SIZE = 50
train_iter = torchtext.data.BucketIterator(traindata,batch_size = BATCH_SIZE,device=device)
test_iter=torchtext.data.BucketIterator(testdata,batch_size = BATCH_SIZE,device=device,shuffle=False)

d_model = 512  # Embedding Size
d_ff = 1024 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 6  # number of heads in Multi-Head Attention
vocab_size = len(vocab1.stoi.keys())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q--(src_len), d_model]
        input_K: [batch_size, len_k--(src_len), d_model]
        input_V: [batch_size, len_v--(=len_k)(src_len), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]

        output = self.fc(context)  # [batch_size, len_q, d_model]


        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]

        return enc_outputs, attn



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.embedding.weight.data.copy_(vocab1.vectors)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc=nn.Linear(300,512)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.embedding(enc_inputs)  # [batch_size, src_len, d_model]，词向量embeding
        enc_outputs=self.fc(enc_outputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1))\
            .transpose(0,1)  # [batch_size, src_len, d_model],生成位置编码，并返回与embeding的加和
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)  # [batch_size, src_len, src_len]，实意词与pad之间没有任何联系，所以对pad进行预处理
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        print(enc_outputs.size())
        return enc_outputs, enc_self_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.projection = nn.Linear(400*d_model,100, bias=False).cuda()
        self.fc=nn.Linear(100,10)
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs,_ = self.encoder(enc_inputs)

        enc_outputs=enc_outputs.view(enc_outputs.size()[0],-1)
        return self.fc(self.projection(enc_outputs))

Transformer_model = Transformer().cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(Transformer_model.parameters(), lr=0.00003)
print(Transformer_model)

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
    for epoch in range(1,num_epochs+1):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
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
            train_num += len(target)
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{} Train Loss: {:.6f}  Train Acc: {:.6f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))
        gc.collect()
        torch.cuda.empty_cache()
train_model1(Transformer_model,train_iter,loss_func,optimizer,num_epochs=2)

result=pd.read_csv(r"C:/Users/NewNLPer/Desktop/za/城堡竞赛/submit_example.csv")
with torch.no_grad():
    for step, batch in enumerate(test_iter):
        textdata, target = batch.txt[0], batch.Label.view(-1)
        out = Transformer_model(textdata.to(device))
        pre_lab = torch.argmax(out, 1)  # 预测的标签
        for i in range(0,50):
            result['Label'][(step*50)+i]=int(pre_lab[i])
            print('预测数据正在写入submit_example，已完成%d'%((step*50)+i))
print()
result[['ID','Label']].to_csv('C:/Users/NewNLPer/Desktop/za/城堡竞赛/submit_example.csv',index=False)
print('----------预测数据已经全部写入，已保存，请查看！！！----------')
