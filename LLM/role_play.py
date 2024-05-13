# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/13 20:30
coding with comment！！！
"""

import os

os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = "sk-**"

from chatharuhi import ChatHaruhi

chatbot = ChatHaruhi( role_from_hf = 'chengli-thu/linghuchong', \
                      llm = 'openai')

response = chatbot.chat(role='小师妹', text = '冲哥。')
print(response)
