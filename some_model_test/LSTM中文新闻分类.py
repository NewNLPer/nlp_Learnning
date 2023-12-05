# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2022/12/8 10:29
coding with comment！！！
"""
###导入需要的库

import pandas as pd
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname = "/Library/Fonts/华文细黑.ttf")## 输出图显示中文
import torch
from torch import nn
import torchtext
from torchtext import vocab
from torchtext.data import Field
import re
import jieba
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
init_token：文本的起始字符，默认为None。
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
2.生成词向量矩阵vocab1.vectors，为每个既在所建立的词表中的，又在预训练词向量里的单词给出数值表示(将预训练里的vector拿过来)，对于不符合这种情况的单词的数值表示全为【0，0，0，...，0】
3.其实有一个很有意思的事情，就是训练集中单词在预训练词表中的数量，这时候就要看所给的预训练词表全不全了，所以预训练的词表还是非常重要的。
'''

vocab1=TEXT.vocab #显示所建立的词典
#基础查询:vocab1.vectors(词向量矩阵),vocab1.stio(词的索引)

# print(vocab1.stoi)
exit()


s=0
for item in vocab1.stoi.keys():
    if item in word_vectors.stoi.keys():
        s+=1
print()
print('训练集词汇表中有%f词语在预训练词向量词表中'%(s/len(vocab1.stoi.keys())))
print()
'============================================================================================='


BATCH_SIZE = 64

train_iter = torchtext.data.BucketIterator(traindata,batch_size = BATCH_SIZE,device=device)
val_iter = torchtext.data.BucketIterator(valdata,batch_size = BATCH_SIZE,device=device)
test_iter = torchtext.data.BucketIterator(testdata,batch_size = BATCH_SIZE,device=device)
#在本条命令之前所给的训练集还是文字，进行batch划分之后，就成了所在词表中的索引了。
#train_iter有三个属性，batch.cutword[0](词的的索引),batch.cutword[1](实词的个数),batch.labelcode(标签)
'''
# Field：用来定义字段以及文本预处理方法
# Example: 用来表示一个样本，通常为“数据+标签”
# TabularDataset: 用来从文件中读取数据，生成Dataset， Dataset是Example实例的集合
# BucketIterator：迭代器，用来生成batch， 类似的有Iterator，BucketIterator的功能较强大点，支持排序，动态padding等
# '''
#  获得一个batch的数据，对数据进行内容进行介绍
for step, batch in enumerate(train_iter):
    c1 = batch.cutword[0][1]
    print("数据的尺寸:",batch.cutword[0].shape)
## 针对一个batch 的数据，可以使用batch.labelcode获得数据的类别标签
# print("数据的类别标签:\n",batch.labelcode)
## batch.cutword[0]是文本对应的标签向量
# print("数据的尺寸:",batch.cutword[0].shape)
## batch.cutword[1] 对应每个batch使用的原始数据中的索引
# print("数据样本数:",batch.cutword[0][1])

exit()

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,batch_first=True)
        self.fc=nn.Linear(hidden_dim,50)
        self.fc1 = nn.Linear(50, output_dim)

    def forward(self, x):
        #x(batch_size,seq_len)--->(batch_size,seq_len,embedding_dim)--->)
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(embeds, None)   # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        out = self.fc(r_out[:, -1, :])
        out=self.fc1(out)
        return out
vocab_size = vocab1.vectors.size(0)###词典的长度
embedding_dim = vocab1.vectors.size(1)###嵌入的维度

hidden_dim = 100
layer_dim = 1
output_dim = 10
lstmmodel = LSTMNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)

lstmmodel=lstmmodel.to(device)
print(lstmmodel)

def train_model2(model, traindataloader, valdataloader, criterion,
                 optimizer, num_epochs=25):
    """
    model:网络模型；
    traindataloader:训练数据集;
    valdataloader:验证数据集;
    criterion：损失函数；
    optimizer：优化方法；
    num_epochs:训练的轮数
    """
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(1,num_epochs+1):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        model.train()  ## 设置模型为训练模式
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
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
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        ## 计算一个epoch的训练后在验证集上的损失和精度
        model.eval()  ##

        # 设置模型为训练模式评估模式
        for step, batch in enumerate(valdataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            out = model(textdata.to(device))
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, target.to(device))
            val_loss += loss.item() * len(target)
            val_corrects += torch.sum(pre_lab == target.data.to(device))
            val_num += len(target)
        ## 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Val Loss: {:.4f}  Val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return model, train_process
optimizer = torch.optim.Adam(lstmmodel.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss()   # 损失函数
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
lstmmodel,train_process = train_model2(
    lstmmodel,train_iter,val_iter,loss_func,optimizer,num_epochs=5)

print('------------------训练结束------------------')
print()
print('------------------开始预测------------------')

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
    finnaly_out=lstmmodel(pre.to(device))
    pre_lab = torch.argmax(finnaly_out, 1)
    dict_list = {"体育": 0, "娱乐": 1, "家居": 2, "房产": 3, "教育": 4,
                 "时尚": 5, "时政": 6, "游戏": 7, "科技": 8, "财经": 9}
    mydict_new = dict(zip(dict_list.values(), dict_list.keys()))
    return mydict_new[int(pre_lab)]


s='1月12日16时30分，世界最高海中大桥——深中通道伶仃洋大桥首榀箱梁成功吊装，标志着大桥建设进入上部结构安装新阶段，大桥建设取得新进展。深中通道连接珠江口东岸的深圳和珠江口西岸的中山，是粤港澳大湾区重要枢纽工程，全长约24公里，是集“桥、岛、隧、水下互通”于一体的跨海集群工程。伶仃洋大桥是深中通道关键控制性工程之一，为主跨1666米的三跨全漂浮体系悬索桥，主塔高270米，是世界最大跨径海中钢箱梁悬索桥、最高海中大桥。深中通道预计2024年实现通车，建成后将有力推进珠三角经济、交通一体化及转型升级，成为联系珠江口东西两岸的直联通道，届时从深圳到中山车程将由原来的2小时缩减至20分钟。'


print(shi_ce(s))





#######总结########
'''
1.各种词向量的特点：
One-hot：维度灾难 and 语义鸿沟
矩阵分解（LSA）：利用全局语料特征，但SVD求解计算复杂度大
基于NNLM/RNNLM的词向量：词向量为副产物，存在效率不高等问题
word2vec、fastText：优化效率高，但是基于局部语料，（可以用gensim来进行训练自己的word2vec）
glove：基于全局语料，结合了LSA和word2vec的优点
elmo、GPT、bert：动态特征
2.如何使用预训练的模型的权重矩阵，nn.embedding的deepcopy，这是一个非常重要的问题。
3.飞浆————国产之光，但是我还是不想用，pytorch肯定有使用的
4.https://huggingface.co/的使用还有待开发
5.cuda的加速，to(device)
'''

