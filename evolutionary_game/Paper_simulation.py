# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/21 14:13
coding with comment！！！
"""
# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/11 19:07
coding with comment！！！
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm



def Cooperation_proportion_derivatives(x, t, yib, sit):
    """
    :param x:  Initial variable[x,n]
    :param t: time
    :return:
    """
    piC = (5 - 2 * x[1]) * x[0] + (1 - x[1]) * (1 - x[0])
    piD = (3 + 2 * x[1]) * x[0] + x[1] * (1 - x[0])

    function_1 = x[0] * (1 - x[0]) * (piC - piD) / yib

    function_2 = x[1] * (1 - x[1]) * ((1 + sit) * x[0] - 1)

    return [function_1, function_2]


def plot_Time_evolution_chart(x,t):
    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]

    plt.plot(t,Collaborator_ratio)
    plt.xlabel('t')
    plt.ylabel('pc')
    plt.title("Collaborator_ratio")
    plt.show()


def plot_variogram(x,b):

    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]

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

    initial_x = [0.8, 0.1]
    yib = 0.1
    sit = 2

    t = np.linspace(0, 50, 1000)

    result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(yib,sit))
    plot_Time_evolution_chart(result,t)
    print(result[-1])









