# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/23 12:42
coding with comment！！！
"""
# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/11 19:07
coding with comment！！！
"""

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import math
import random


initial_x = [0.5, 0.5]
t = list(range(1, 15001))
alph = 0.2
yit = 0.1
sit = 0.1
cim = 2

k_1 = 0.8
k_2 = 0.5

def Cooperation_proportion_derivatives(x, t, b, alph, yit,sit):
    return_ = """
    :param x:  Initial variable[x,M]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :return:
    """
    piC = x[0] * (1 + k_1 * (1 - x[1]) ** cim)

    piD = b * x[0] * (1 - k_2 * x[1] ** cim)

    function_1 = x[0] * (1 - x[0]) * (piC - piD)

    function_2 = sit * x[1] * (1 - x[1]) * (alph * x[0] - yit * x[1])

    return [function_1, function_2]


def plot_Time_evolution_chart(x,t): # 单一变量的 时间演化图
    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]

    plt.plot(t,Collaborator_ratio)
    plt.xlabel('t')
    plt.ylabel('pc')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(t,Degree_of_rewards_and_punishments)
    plt.xlabel('t')
    plt.ylabel('atm')
    plt.title("Degree_of_Atmosphere")
    plt.show()


def plot_variogram(x,b): #单一变量的 背叛诱惑图

    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]


    # 绘制折线图，同时指定线条颜色
    plt.plot(b, Collaborator_ratio, label='Pc', color='red')  # 红色线条
    plt.plot(b, Degree_of_rewards_and_punishments, label='M', color='green')  # 绿色线条
    # 添加五角星标记
    plt.scatter(b, Collaborator_ratio, marker='*', color='red')
    plt.scatter(b, Degree_of_rewards_and_punishments, marker='*', color='green')

    # 添加其他元素
    plt.legend()
    plt.title('Collaborator_ratio and Degree_of_Atmosphere')
    plt.xlabel('b')
    plt.ylabel('y')
    # 显示图形
    plt.show()


def linespace(start,end,interval): # 为防止精度溢出，定义间隔

    float_lens = len(str(interval).split(".")[-1])
    save_list = []
    while start <= end:
        save_list.append(start)
        start += interval
        start = round(start,float_lens)
    if save_list[-1] != end:
        save_list.append(end)
    return save_list


def time_evloution(b):  # 单一变量的 时间演化图

    result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b,alph,yit,sit))
    plot_Time_evolution_chart(result,t)


def single_plot(alph,yit,state):
    """
    :param alph:
    :param bit:
    :param state: 0 画图 1 返回结果
    :return:
    """
    if state:
        result_finally = []
        line_space_b = linespace(1,1.99,0.05)
        for b in tqdm(line_space_b):
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b,alph,yit,sit))
            result_finally.append(get_round(result[-1].tolist()))
        plot_variogram(result_finally, line_space_b)
    else:
        result_finally = []
        line_space_b = linespace(1, 1.99, 0.05)
        for b in tqdm(line_space_b):
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit))
            result_finally.append(get_round(result[-1].tolist()))
        return [line_space_b,result_finally]



def get_round(list):
    return [round(item,3) for item in list]


def multivariable_plot(variable):
    """
    :param variable:[(a1,b1),(a2,b2),(a3,b3),...]
    :return:
    """

    polt_dic={}

    for item in variable:
        alph = "alph = {}".format(item[0])
        bit = "yit = {}".format(item[1])
        result = single_plot(item[0],item[1],0)
        polt_dic[alph + " * " + bit] = [result[0],[sublist[0] for sublist in result[1]],[sublist[1] for sublist in result[1]]]

    color_set = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    for key in polt_dic: # 绘制合作者比例随背叛诱惑的变化图

        nums = random.randint(0,len(color_set) - 1)
        plt.plot(polt_dic[key][0], polt_dic[key][1], label=key, color=color_set[nums])  # 红色线条
        plt.scatter(polt_dic[key][0], polt_dic[key][1], marker='*', color='red')
        color_set = color_set[:nums] + color_set[nums+1:]


    plt.legend()
    plt.title('Collaborator_ratio')
    plt.xlabel('b')
    plt.ylabel('pc')
    plt.show()

    color_set = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    for key1 in polt_dic: # 绘制氛围M随背叛诱惑的变化图

        nums = random.randint(0,len(color_set) - 1)
        plt.plot(polt_dic[key1][0], polt_dic[key1][2], label=key1, color=color_set[nums])  # 红色线条
        plt.scatter(polt_dic[key1][0], polt_dic[key1][2], marker='*', color='red')
        color_set = color_set[:nums] + color_set[nums+1:]

    plt.legend()
    plt.title('Degree_of_Atmosphere')
    plt.xlabel('b')
    plt.ylabel('y')
    plt.show()



if __name__=="__main__":


    # time_evloution(1.8)

    # print(single_plot(alph,bit))

    for item in [0.1,0.4,0.7]:
        multivariable_plot([(0.1,item),(0.3,item),(0.5,item),(0.7,item),(0.9,item)])










