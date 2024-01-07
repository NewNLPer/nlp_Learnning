# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/1/2 11:33
coding with comment！！！
"""


import dashscope
dashscope.api_key="sk-9e53478e820141c9b472154e45b6a1be"

def sample_sync_call_streaming():
    prompt_text = '请简单帮我介绍下llama模型'
    response_generator = dashscope.Generation.call(
        model='qwen-max',
        prompt=prompt_text,
        stream=True,
        top_p=0.8)

    head_idx = 0
    for resp in response_generator:
        paragraph = resp.output['text']
        print("\r%s" % paragraph[head_idx:len(paragraph)], end='')
        if(paragraph.rfind('\n') != -1):
            head_idx = paragraph.rfind('\n') + 1

sample_sync_call_streaming()