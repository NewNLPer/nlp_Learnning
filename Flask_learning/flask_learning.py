# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/20 21:12
coding with comment！！！
"""
from flask import Flask
flask_learning = Flask(__name__)

@flask_learning.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    flask_learning.run()