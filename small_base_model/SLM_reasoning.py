# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/13 17:09
coding with comment！！！
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


model_path="D:\Test_model_weights"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# start_time=time.time()
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")
model.cuda()
# emd_time=time.time()
# print("consume time {}".format(emd_time-start_time))


input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")["input_ids"].cuda()
# print(inputs)


outputs = model.generate(inputs)[0]
print(tokenizer.decode(outputs))



