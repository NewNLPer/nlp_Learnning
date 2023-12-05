# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/1/20 11:17
coding with comment！！！
"""
import pandas as pd
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname = "/Library/Fonts/华文细黑.ttf")## 输出图显示中文
import torch
from torch import nn
import torchtext
from torchtext import vocab
from torchtext.data import Field
import math
import re
import jieba
import numpy as np
import torch.optim as optim
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
    ("labelcode", LABEL), # 对标签的操作
    ("cutword", TEXT) # 对文本的操作
]
"""
sequential=True:表明输入的文本时字符，而不是数值字
tokenize="spacy":使用spacy切分词语
use_vocab=True: 创建一个词汇表
batch_first=True: batch优先的数据方式
fix_length=400 :每个句子固定长度为400
——————————————————————————————————————————————————————
更加详细的介绍如下：
squential：数据是否为序列数据，默认为Ture。如果为False，则不能使用分词。
use_vocab：是否使用词典，默认为True。如果为False，那么输入的数据类型必须是数值类型(即使用vocab转换后的)。
init_token：文本的其实字符，默认为None。
eos_token：文本的结束字符，默认为None。
fix_length：所有样本的长度，不够则使用pad_token补全。默认为None，表示灵活长度。
tensor_type：把数据转换成的tensor类型 默认值为torch.LongTensor。
preprocessing：预处理pipeline， 用于分词之后、数值化之前，默认值为None。
postprocessing：后处理pipeline，用于数值化之后、转换为tensor之前，默认为None。
lower：是否把数据转换为小写，默认为False；
tokenize：分词函数，默认为str.split
include_lengths：是否返回一个已经补全的最小batch的元组和和一个包含每条数据长度的列表，默认值为False。
batch_first：batch作为第一个维度；
pad_token：用于补全的字符，默认为<pad>。
unk_token：替换袋外词的字符，默认为<unk>。
pad_first：是否从句子的开头进行补全，默认为False；
truncate_first：是否从句子的开头截断句子，默认为False；
stop_words：停用词；
"""
## 读取数据,将其都转换成实例，维度变化，补全长度
traindata,valdata,testdata = torchtext.data.TabularDataset.splits(
    path="D:/charm/pytorch数据/程序/programs/data/chap7", format="csv",
    train="cnews_train2.csv", fields=text_data_fields,
    validation="cnews_val2.csv",
    test = "cnews_test2.csv", skip_header=True
)
"""
对于词表的处理是最重要的地方，需要强调一下:
1.不使用中文预训练词向量
可以通过训练集来创建词表，TEXT.build_vocab(traindata,max_size=20000,vectors=None)
注意这里的vectors=None,当然nn.embeding层初始化权重就可以。
2.使用中文预训练词向量
预训练词向量需要为txt格式进行读取，对于少部分的iters(迭代器格式)可以选择将另存为txt格式，
from gensim.models.keyedvectors import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format("D:\charm\中文预训练词向量\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5.bz2", binary=False,unicode_errors='ignore')
w2v_model.save_word2vec_format('token_vec_300.txt', binary=False)
然后再进行处理，可以以本模型为案例来进行学习。另外nn.embeding需要deep_copy所建立的vocal1.vectors
"""
word_vectors = vocab.Vectors(r"D:\charm\day\NewNLPer\中文预训练词向量\zuihou.txt")###预训练词向量加载
#基础查询:word_vectors.vectors(词向量矩阵),word_vectors.stio(词的索引)
TEXT.build_vocab(traindata,vectors=word_vectors)
'''
TEXT.build_vocab函数的作用:
1.建立训练集的词表，
2.生成词向量矩阵vocab1.vectors，为每个既在所建立的词表中的，又在预训练词向量里的单词给出数值表示，对于不符合这种情况的单词的数值表示全为【0，0，0，...，0】
3.其实有一个很有意思的事情，就是训练集中单词在预训练词表中的数量，这时候就要看所给的预训练词表全不全了，所以预训练的词表还是非常重要的。
'''

vocab1=TEXT.vocab #显示所建立的词典
#基础查询:vocab1.vectors(词向量矩阵),vocab1.stio(词的索引)

s=0
for item in vocab1.stoi.keys():
    if item in word_vectors.stoi.keys():
        s+=1
print()
print('训练集词汇表中有%f词语在预训练词向量词表中'%(s/len(vocab1.stoi.keys())))
print()
####模型的定义
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
vocab_size = len(vocab1.stoi.keys())



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0,1)  # [batch_size, src_len, d_model],生成位置编码，并返回与embeding的加和
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)  # [batch_size, src_len, src_len]，实意词与pad之间没有任何联系，所以对pad进行预处理
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
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

model = Transformer().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
####模型的定义
BATCH_SIZE = 36
train_iter = torchtext.data.BucketIterator(valdata,batch_size = BATCH_SIZE,device=device)
loss11=[]
for i in range(1,31):
    loss1=0
    p=0
    for step,batch in enumerate(train_iter):
        textdata, target = batch.cutword[0], batch.labelcode.view(-1)
        pre=model(textdata)
        loss = criterion(pre, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1+=loss.item()
        p+=textdata.size(0)
    loss11.append(loss1/p)
    print('第%d次迭代，loss为：%f'%(i,loss1/p))
for item in loss11:
    print(item)
from matplotlib import pyplot as plt
plt.plot(loss11)
plt.show()



def generate(str):
    def chinese_pre(text_data):
        stop_words = pd.read_csv("D:/charm/pytorch数据/程序/programs/data/chap7/cnews/中文停用词库.txt", sep="\t",
                             header=None, names=["text"])
        # text_data = text_data.lower()    ## 字母转化为小写
        text_data = re.sub("\d+", "", text_data)   ## 去除数字，正则表达是的强大之处
        text_data = list(jieba.cut(text_data,cut_all=False))      ## 分词,使用精确模式(值得注意的是，jieba自动采用HMM来进行计算最大概率，从而实现分词)
        text_data = [word.strip() for word in text_data if word not in stop_words.text.values]     ## 去停用词和多余空格
        text_data = " ".join(text_data) ## 处理后的词语使用空格连接为字符串
        text_data=text_data.split(' ')
        return text_data
    index1 = []
    for item in chinese_pre(str):
        if len(index1)<400:
            index1.append(vocab1.stoi.get(item, 0))
        else:
            break
    for i in range(len(index1),400):
        index1.append(1)
    return index1

def shi_ce(s):
    pre=torch.tensor(generate(s))
    pre=torch.unsqueeze(pre,dim=0)
    finnaly_out=model(pre.to(device))
    pre_lab = torch.argmax(finnaly_out, 1)
    dict_list = {"体育": 0, "娱乐": 1, "家居": 2, "房产": 3, "教育": 4,
                 "时尚": 5, "时政": 6, "游戏": 7, "科技": 8, "财经": 9}
    mydict_new = dict(zip(dict_list.values(), dict_list.keys()))
    return mydict_new[int(pre_lab)]

s='1月12日16时30分，世界最高海中大桥——深中通道伶仃洋大桥首榀箱梁成功吊装，标志着大桥建设进入上部结构安装新阶段，大桥建设取得新进展。深中通道连接珠江口东岸的深圳和珠江口西岸的中山，是粤港澳大湾区重要枢纽工程，全长约24公里，是集“桥、岛、隧、水下互通”于一体的跨海集群工程。伶仃洋大桥是深中通道关键控制性工程之一，为主跨1666米的三跨全漂浮体系悬索桥，主塔高270米，是世界最大跨径海中钢箱梁悬索桥、最高海中大桥。深中通道预计2024年实现通车，建成后将有力推进珠三角经济、交通一体化及转型升级，成为联系珠江口东西两岸的直联通道，届时从深圳到中山车程将由原来的2小时缩减至20分钟。'

print(shi_ce(s))

