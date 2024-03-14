# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/13 16:48
coding with comment！！！
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, datetime
import torch
import os
from transformers import AutoModel, AutoTokenizer
from transformers.generation.utils import GenerationConfig
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"

app = FastAPI()

class Query(BaseModel):
    text: str


model_path = "/data/ubuntu/public/models/chatglm_6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
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
    uvicorn.run(app, host="10.1.0.68", port=1022)
   #  import requests
   #  url="http://10.1.0.68:1022/chat/"
   #  query ={"text":"你是谁？"}
   #  response =requests.post(url,json=query)
   #  if response.status_code == 200:
   #     result =response.json()
   #     print("BOT:",result["result"])
   # else:
   #     print("Error:",response.status_code, response.text)


print(input())