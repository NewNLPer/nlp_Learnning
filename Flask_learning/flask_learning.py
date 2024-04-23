# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/20 21:12
coding with comment！！！
"""
from flask import Flask

app = Flask(__name__)

# 定义路由
@app.route('/')
def home():
    return 'Hello, this is the home page!'

@app.route('/about')
def about():
    return 'This is the about page.'

# 运行应用
if __name__ == '__main__':

    app.run(host = "10.5.159.149",port=5910)