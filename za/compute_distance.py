# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/22 12:10
coding with comment！！！
"""

import json
import requests

# 申请的高德的Api-key
key = "8d8b55e388a180c9c7913e8f5e8ab10b"

"""
更详细的信息可参考官方技术文档，https://lbs.amap.com/api/webservice/guide/api/direction#bus
"""

def get_lon_lat(i):
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    parameters = {
        'key':key,
        'address':'%s' % i
        }
    page_resource = requests.get(url,params=parameters)
    text = page_resource.text
    data = json.loads(text)
    lon_lat = data["geocodes"][0]['location']
    return lon_lat

def routes(origin,destination):
    origin  = get_lon_lat(origin)
    destination = get_lon_lat(destination)
    parameters = {'key':key,'origin':origin,'destination':destination}
    response = requests.get('https://restapi.amap.com/v3/direction/driving?parameters',params=parameters)
    text = json.loads(response.text)
    duration = text['route']['paths'][0]['duration']
    # 换算成小时
    return str(round(int(duration)/3600,2)) + "h"

if __name__ == "__main__":

    city_1 = "日照"
    city_2 = "青岛"
    traval_time_sum = routes(city_1,city_2)
    print("从{}到{}驾车需要{}".format(city_1,city_2,traval_time_sum))



