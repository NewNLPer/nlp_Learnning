
# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/4/1 15:10
coding with comment！！！
"""
import sys
from datetime import datetime
from os.path import join
from tqdm.auto import tqdm, trange
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config
import numpy as np

'''
I:最近天气好好，想出去拍照片
Y:我也有这个打算，不过我的相机是渣渣
I:哈哈哈我也不是专业的，我就是瞎拍，多拍拍就好了
Y:那你打算什么时候去拍啊
I:就这两天吧，刚好是清明节
Y:嗯，我也差不多，清明后就要开始忙了
I:我是学生所以还好哈哈哈，平时间都比较多的
Y:羡慕，我都已经毕业了
I:我也想快点毕业，然后赚钱
Y:毕业之后你就会怀念上学的时候，还是好好珍惜当下的生活吧
I:哈哈哈，每个阶段都会怀念过去
Y:嗯，人都是这样的，所以要好好的珍惜现在
I:对啊，人生无常，过好每一天
Y:你这么年纪轻轻就参悟生命的真谛哈哈哈

数据转化成下面类型：

CLS最近天气好好，想出去拍照片SEP我也有这个打算，不过我的相机是渣渣SEP哈哈哈我也不是专业的，
我就是瞎拍，多拍拍就好了SEP那你打算什么时候去拍啊SEP就这两天吧，刚好是清明节SEP嗯，
我也差不多，清明后就要开始忙了SEP我是学生所以还好哈哈哈，平时间都比较多的SEP羡慕，
我都已经毕业了SEP我也想快点毕业，然后赚钱SEP毕业之后你就会怀念上学的时候，
还是好好珍惜当下的生活吧SEP哈哈哈，每个阶段都会怀念过去SEP嗯，人都是这样的，
所以要好好的珍惜现在SEP对啊，人生无常，过好每一天SEP你这么年纪轻轻就参悟生命的真谛哈哈哈SEP

'''























