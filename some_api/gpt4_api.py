# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/1/2 12:21
coding with comment！！！
"""
import openai
from IPython.display import Markdown, display

# 设置OpenAI API密钥
openai.api_key = "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"

system_intel = "You are GPT-4, answer my questions as if you were an expert in the field."
prompt = "Write a blog on how to use GPT-4 with python in a jupyter notebook"

# 调用GPT-4 API的函数
def ask_GPT4(system_intel, prompt):
    result = openai.ChatCompletion.create(model="gpt-4",
                                 messages=[{"role": "system", "content": system_intel},
                                           {"role": "user", "content": prompt}])
    display(Markdown(result['choices'][0]['message']['content']))

# 调用函数
ask_GPT4(system_intel, prompt)