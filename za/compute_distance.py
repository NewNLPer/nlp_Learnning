# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/22 12:10
coding with comment！！！
"""

import json
import requests


key = "8d8b55e388a180c9c7913e8f5e8ab10b"


def get_lon_lat(i):
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    parameters = {
        'key':key,   							##输入自己的key
        'address':'%s' % i
        }
    page_resource = requests.get(url,params=parameters)
    text = page_resource.text       ##获得数据是json格式
    data = json.loads(text)         ##把数据变成字典格式
    lon_lat = data["geocodes"][0]['location']
    return lon_lat

def routes(origin,destination):
    origin  = get_lon_lat(origin)
    destination = get_lon_lat(destination)
    parameters = {'key':key,'origin':origin,'destination':destination}
    ##参数的输入，可以按照自己的需求选择出行时间最短，出行距离最短，不走高速等方案，结合自己需求设置，参考手册
    response = requests.get('https://restapi.amap.com/v3/direction/driving?parameters',params=parameters)
    text = json.loads(response.text)
    # print(text)
    duration = text['route']['paths'][0]['duration'] ##出行时间
    ## 可以自己打印text看一下，能提取很多参数，出行时间、出行费用、出行花费等看自己需求提取

    return duration/3600


