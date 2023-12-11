# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/11 19:07
coding with comment！！！
"""
# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/6 14:52
coding with comment！！！
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import time
from tqdm import tqdm



def Cooperation_proportion_derivatives(x, t, punish, b, xi, p_1, p_2):
    """
    :param x:  Initial variable[x,ac,ad]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :return:
    """
    piC = x[0] - x[1] * punish * (1 - x[0])
    piD = (b - x[2] * punish) * x[0]
    function_1 = x[0] * ( 1 - x[0] ) * (piC - piD)
    function_2 = xi * x[1] * ( 1 - x[1] ) * (p_1 * piD - piC)
    function_3 = xi * x[2] * ( 1 - x[2] ) * (p_2 * piC - piD )
    return [function_1, function_2, function_3]

def get_plot(x,t,remark):

    Collaborator_ratio= [sublist[0] for sublist in x]
    C_Perceived_degree=[sublist[1] for sublist in x]
    D_Perceived_degree = [sublist[2] for sublist in x]

    # plt.figure(figsize=(12, 20))

    # plt.subplot(3, 1, 1)
    plt.plot(t,Collaborator_ratio)
    plt.xlabel('b')
    plt.ylabel('c')
    plt.title("Collaborator_ratio")
    plt.show()

    # plt.subplot(3, 1, 2)
    plt.plot(t,C_Perceived_degree)
    plt.xlabel('b')
    plt.ylabel('ac')
    plt.title("C_Perceived_degree")
    plt.show()

    # plt.subplot(3, 1, 3)
    plt.plot(t,D_Perceived_degree)
    plt.xlabel('b')
    plt.ylabel('ad')
    plt.title("D_Perceived_degree")
    plt.show()

    # plt.suptitle("Parameter settings {}".format(remark))
    # plt.savefig(r'C:/Users/NewNLPer/Desktop/za/exp_figure/{}.png'.format(remark))
    # plt.show()



def get_remark(b,punish,xi,p_1,p_2):
    remark="b={}_punish={}_xi={}_p1={}_p2={}".format(b,punish,xi,p_1,p_2)
    return remark


def linespace(start,end,interval):

    float_lens=len(str(interval).split(".")[-1])
    save_list=[]
    while start<=end:
        save_list.append(start)
        start+=interval
        start=round(start,float_lens)
    if save_list[-1] != end:
        save_list.append(end)
    return save_list

def get_round(list):
    for i in range(len(list)):
        list[i]=round(list[i],3)
    return list


if __name__=="__main__":

    initial_x=[0.5,0.5,0.5]
    t = np.linspace(0,2000,2000)
    punish = 0.1
    xi = 0.01
    p_1 = 0.5
    p_2 = 1.5

    result_=[]

    line_space_b=linespace(1,2,0.0001)
    for b in tqdm(line_space_b):

        result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(punish, b, xi,p_1,p_2))
        result_.append(get_round(result[-1].tolist()))

    get_plot(result_,line_space_b," ")


    # get_plot(result,t,remark)






