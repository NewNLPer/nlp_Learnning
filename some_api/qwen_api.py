# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/1/2 11:33
coding with comment！！！
"""


import dashscope
dashscope.api_key="sk-9e53478e820141c9b472154e45b6a1be"

def sample_sync_call_streaming():
    prompt_text = 'Q: In this task, you are given two phrases: Head and Tail, separated with <sep>. The Head and the Tail events are short phrases possibly involving participants. The names of specific people have been replaced by generic words (e.g., PersonX, PersonY, PersonZ). PersonX is always the subject of the event. You have to determine whether the Head is located or can be found at/in/on the Tail or not. Classify your answers into \"Yes\" and \"No\". The phrase may also contain \"___\", a placeholder that can be an object, a person, and/or an action.\nHead: chicken<sep>Tail: plate\nA:'
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