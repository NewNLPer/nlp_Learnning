# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/23 12:42
coding with comment！！！
"""

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm



epsilon = 0.1
alpha = 0.1
beta = 0.01
theta = 0.1
mu = 0.1

def Cooperation_proportion_derivatives(x, t, b):
    """
    :param x:  Initial variable[x,M,H]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :return:
    """
    piC = x[0]

    piD = b * x[0]

    function_1 = x[0] * (1 - x[0]) * ( (1 + x[1]) * piC - (1 - x[1]) * piD + (x[1] - 0.5) * (x[1] - 0.5))

    function_2 = epsilon * x[1] * (1 - x[1]) * (alpha * (2 * x[0] - 1) - beta * x[1] * x[1] + mu * (2 * x[2] - 1))

    function_3 = theta * (x[0] - x[2])

    return [function_1, function_2, function_3]


def plot_Time_evolution_chart(x,t):
    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]
    lishi = [sublist[2] for sublist in x]



    plt.plot(t,Collaborator_ratio)
    plt.xlabel('t')
    plt.ylabel('pc')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(t,Degree_of_rewards_and_punishments)
    plt.xlabel('t')
    plt.ylabel('degree')
    plt.title("Degree_of_rewards_and_punishments")
    plt.show()

    plt.plot(t,lishi)
    plt.xlabel('t')
    plt.ylabel('degree')
    plt.title("Degree_of_rewards_and_punishments")
    plt.show()


def plot_variogram(x,b):

    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]
    lishi = [sublist[2] for sublist in x]

    plt.plot(b,Collaborator_ratio)
    plt.xlabel('b')
    plt.ylabel('pc')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(b,Degree_of_rewards_and_punishments)
    plt.xlabel('b')
    plt.ylabel('degree')
    plt.title("Degree_of_rewards_and_punishments")
    plt.show()

    plt.plot(b,lishi)
    plt.xlabel('b')
    plt.ylabel('degree')
    plt.title("Degree_of_rewards_and_punishments")
    plt.show()


def linespace(start,end,interval):

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

if __name__=="__main__":
    initial_x = [0.5, 0.1, 0.1]
    t = list(range(1,10001))


    # # 1. 固定背叛诱惑b的时间演化图
    # b = 1.5
    #
    # result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b,sit,yit,hit))
    # plot_Time_evolution_chart(result,t)


   # 2. 背叛诱惑b变量的演化图
    result_finally = []
    line_space_b = linespace(1,2,0.001)
    for b in tqdm(line_space_b):
        result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b,))
        result_finally.append(get_round(result[-1].tolist()))
    plot_variogram(result_finally, line_space_b)








