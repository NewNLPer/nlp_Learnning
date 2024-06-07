# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/6/7 22:31
coding with comment！！！
"""


from flask import Flask, send_file, request
import logging
from datetime import datetime

app = Flask(__name__)

# 配置日志记录
logging.basicConfig(level=logging.INFO)

@app.route('/download', methods=['GET'])
def download_file():
    # 获取客户端的IP地址
    client_ip = request.remote_addr
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 打印日志
    app.logger.info(f"File requested from IP: {client_ip} at {current_time}")

    # 替换为你要提供下载的文件路径
    file_path = r"C:\Users\NewNLPer\Desktop\0607毕业照.rar"
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, host='10.25.8.53', port=5910)

