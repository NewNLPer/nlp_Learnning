# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/1/18 12:58
coding with comment！！！
"""
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn

model_path = r"C:\Users\NewNLPer\Desktop\bert"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

model = AutoModel.from_pretrained(model_path, config=config)
print(model)

class pooler(nn.Module):
    def __init__(self,):
        super(pooler, self).__init__()
        self.dense = nn.Linear(768,768)
        self.activation = nn.Tanh()
    def forward(self,x):
        return self.activation(self.dense(x))

new_pool = pooler()
model.pooler = new_pool





























# from transformers import AutoTokenizer, AutoModel, AutoConfig
# import torch
# import torch.nn as nn
#
# class RotationalPositionalEncoding(nn.Module):
#     def __init__(self, max_position_embeddings, embedding_size):
#         super(RotationalPositionalEncoding, self).__init__()
#         self.embedding_size = embedding_size
#         self.max_position_embeddings = max_position_embeddings
#         self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
#
#     def forward(self, input_ids):
#         position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         return position_embeddings
#
# # 路径到您的BERT模型
# model_path = r"C:\Users\NewNLPer\Desktop\bert"
#
# # 1. 加载 tokenizer 和配置
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# config = AutoConfig.from_pretrained(model_path)
#
# # 2. 加载模型
# bert_model = AutoModel.from_pretrained(model_path, config=config)
# print(bert_model)


# 3. 替换位置编码为旋转位置编码
# rotational_position_encoding = RotationalPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#
#
# # print(rotational_position_encoding)
#
# bert_model.embeddings.position_embeddings = rotational_position_encoding
# print(bert_model)



# 现在，bert_model 将使用旋转位置编码。您可以继续使用它进行进一步的训练或分析。
