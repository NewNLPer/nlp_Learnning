# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/2/26 16:10
coding with comment！！！
"""



import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import random
import numpy as np


def Cooperation_proportion_derivatives(x, t, b, alph, yit,sit,k_1,k_2):
    return_ = """
    :param x:  Initial variable[x,M]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :k_1:奖励程度
    :k_2:惩罚力度
    :return:
    """
    piC = x[0] * (1 + k_1 * (1 - x[1]) ** 2)
    piD = b * x[0] * (1 - k_2 * x[1] ** 2)
    function_1 = x[0] * (1 - x[0]) * (piC - piD)
    function_2 = sit * x[1] * (1 - x[1]) * (alph * x[0] - yit * x[1])

    return [function_1, function_2]



if __name__=="__main__":

    initial_x = [0.5, 0.5]
    t = list(range(1, 100001))
    alph = 0.1
    yit = 0.1
    sit = 0.1
    k_1 = 0.5
    k_2 = 0.5
    b = 1.6
    k1 = np.arange(0, 1.01, 0.01)
    k2 = np.arange(0, 1.01, 0.01)
    data_pc = np.zeros((len(k1), len(k1)))
    data_m = np.zeros((len(k1), len(k1)))
    tem_b = np.arange(1, 2.01, 0.01)


    for i in tqdm(range(len(k1))):
        for j in range(len(k2)):
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit, k1[i], k2[j]))
            data_pc[i][j]=result[-1][0]
            data_m[i][j] = result[-1][-1]

    plt.pcolormesh(k2,k1, data_pc, cmap='viridis')
    # 添加颜色条
    plt.colorbar()
    # 添加标签和标题
    plt.xlabel('$k_{2}$')
    plt.ylabel('$k_{1}$')
    plt.title("$b=%s$"%(b))
    # 显示图形
    plt.show()

    plt.pcolormesh(k2,k1, data_m, cmap='viridis')
    # 添加颜色条
    plt.colorbar()
    # 添加标签和标题
    plt.xlabel('$k_{2}$')
    plt.ylabel('$k_{1}$')
    plt.title("$b=%s$"%(b))
    # 显示图形
    plt.show()