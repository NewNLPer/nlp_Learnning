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


def Cooperation_proportion_derivatives(x, t, punish, b, xi):
    """
    :param x:  Initial variable[x,pr]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :return:
    """

    piC = x[0] + x[1] * punish * (1 - x[0])
    piD = (b - x[1] * punish) * x[0]

    function_1 = x[0] * (1 - x[0]) * (piC - piD)
    function_2 = xi * x[1] * (1 - x[1]) * (piD - piC)

    return [function_1, function_2]

def get_plot(x,t):

    Collaborator_ratio = [sublist[0] for sublist in x]
    Degree_of_rewards_and_punishments = [sublist[1] for sublist in x]

    plt.plot(t,Collaborator_ratio)
    plt.xlabel('b')
    plt.ylabel('pc')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(t,Degree_of_rewards_and_punishments)
    plt.xlabel('b')
    plt.ylabel('degree')
    plt.title("Degree_of_rewards_and_punishments")
    plt.show()



    # plt.suptitle("Parameter settings {}".format(remark))
    # plt.savefig(r'C:/Users/NewNLPer/Desktop/za/exp_figure/{}.png'.format(remark))
    # plt.show()


def get_remark(b,punish,xi):
    remark="b={}_punish={}_xi={}".format(b,punish,xi)
    return remark



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

    initial_x=[0.5,0]
    t = np.linspace(0,2000,2000)
    punish = 0.1
    xi = 0.01

    result_=[]
    line_space_b=linespace(1,2,0.0001)

    result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(punish, xi))

    for b in tqdm(line_space_b):

        result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(punish, b, xi))
        result_.append(get_round(result[-1].tolist()))

    get_plot(result_,line_space_b)


    # get_plot(result,t,remark)






