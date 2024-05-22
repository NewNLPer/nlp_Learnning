# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/13 16:48
coding with comment！！！
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
app = FastAPI()

class Query(BaseModel):
    text: str

model_path = "/home/ubuntu/model/baichuan-inc/Baichuan2-7B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).cuda()
model = model.eval()

@app.post("/chat/")
async def chat(query: Query):
    input_ids = tokenizer([query.text]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1,
        max_new_tokens=1024)
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    return {"result": outputs}


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.32.23", port=5910)

   #  import requests
   #  url="http://192.168.32.23:5910/chat/"
   #  query ={"text":Final_prompt}
   #  response =requests.post(url,json=query)
   #  if response.status_code == 200:
   #     result =response.json()
   #     print("BOT:",result["result"])
   # else:
   #     print("Error:",response.status_code, response.text)

