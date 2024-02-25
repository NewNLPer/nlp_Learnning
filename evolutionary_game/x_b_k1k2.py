# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/2/25 18:00
coding with comment！！！
"""


import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import math
import random



initial_x = [0.5, 0.5]
t = list(range(1, 200001))
alph = 0.1
yit = 0.1
sit = 0.1
k_1 = 1
k_2 = 1


def Cooperation_proportion_derivatives(x, t, b, alph, yit,sit,k_1,k_2):
    return_ = """
    :param x:  Initial variable[x,M]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :return:
    """
    piC = x[0] * (1 + k_1 * (1 - x[1]) ** 2)
    piD = b * x[0] * (1 - k_2 * x[1] ** 2)
    function_1 = x[0] * (1 - x[0]) * (piC - piD)
    function_2 = sit * x[1] * (1 - x[1]) * (alph * x[0] - yit * x[1])
    return [function_1, function_2]





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


def get_round(list):
    return [round(item,3) for item in list]


def single_plot(k_1,k_2):
    """
    :param alph:
    :param bit:
    :param state: 0 画图 1 返回结果
    :return:
    """
    result_finally = []
    line_space_b = linespace(1, 1.99, 0.05)
    for b in tqdm(line_space_b):
        result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit,k_1,k_2))
        result_finally.append(get_round(result[-1].tolist()))
    return [line_space_b,result_finally]




def plt_k1_k2_figure(k_1,k_2):
    if k_1 == -1: # k_1 = [0,0.3,0.7,1]
        line_space_b = linespace(1, 1.99, 0.05)
        for k_1 in [0,0.3,0.7,1]:
            result = single_plot(k_1,k_2)
            pc = [item[0] for item in result[-1]]
            plt.xlabel('b')
            plt.ylabel(r'$\rho_c$')
            plt.plot(line_space_b, pc, label="$k_{1}=%s$" % (k_1),marker='*')
        plt.legend()
        plt.title('$k_{2}=0.5$')
        plt.show()

        for k_1 in [0,0.3,0.7,1]:
            result = single_plot(k_1,k_2)
            pc = [item[1] for item in result[-1]]
            plt.xlabel('b')
            plt.ylabel('M')
            plt.plot(line_space_b, pc, label="$k_{1}=%s$" % (k_1),marker='*')
        plt.legend()
        plt.title('$k_{2}=0.5$')
        plt.show()

    if k_2 == -1: # k2 = [0,0.3,0.7,1]
        line_space_b = linespace(1, 1.99, 0.05)
        for k_2 in [0,0.3,0.7,1]:
            result = single_plot(k_1,k_2)
            pc = [item[0] for item in result[-1]]
            plt.xlabel('b')
            plt.ylabel(r'$\rho_c$')
            plt.plot(line_space_b, pc, label="$k_{2}=%s$" % (k_2),marker='*')
        plt.legend()
        plt.title('$k_{1}=0.5$')
        plt.show()

        for k_2 in [0,0.3,0.7,1]:
            result = single_plot(k_1,k_2)
            pc = [item[-1] for item in result[-1]]
            plt.xlabel('b')
            plt.ylabel('M')
            plt.plot(line_space_b, pc, label="$k_{2}=%s$" % (k_2),marker='*')
        plt.legend()
        plt.title('$k_{1}=0.5$')
        plt.show()


if __name__=="__main__":
    plt_k1_k2_figure(-1,0.5)