# # -*- coding: utf-8 -*-
# """
# @author: NewNLPer
# @time: 2023/5/5 9:30
# coding with comment！！！
# """
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# from transformers import BertTokenizer, BertModel,AutoTokenizer
# model_name_or_path = r"C:\Users\NewNLPer\Desktop\model"
# tokenizer_name_or_path = r"C:\Users\NewNLPer\Desktop\model"
#
# model=BertModel.from_pretrained(model_name_or_path)
# tokenize=AutoTokenizer.from_pretrained(tokenizer_name_or_path)
#
# # for named_para,weight in model.named_parameters():
# #     print(named_para)
# # exit()
#
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
#     target_modules=['attention.self.query','attention.self.value','attention.self.key']
# )
# model = get_peft_model(model, peft_config)
# trained_
# nums=model.print_trainable_parameters()
# print(model)
#
# text='我喜欢吃苹果'
# text1='哈皮的'
#
# input1=tokenize(text,return_tensors='pt')
# print(model(**input1))
import torch
dic_c={'label':[1,2,3]}
print(dic_c.cuda())
