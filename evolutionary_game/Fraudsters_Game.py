# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/23 14:47
coding with comment！！！
"""


import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm


def Cooperation_proportion_derivatives(x, t, b,sit):
    """
    :param x:  Initial variable[x,y,z,alph,bit]
    :param t: time
    :param punish:
    :param b: b=[1,2]
    :param xi:Growth rate control
    :return:
    """
    piC = x[0] + (1 - x[3]) * x[2]

    piD = b * x[0] + (b - x[-1]) * x[2]

    piP = x[3] * x[0] +x[-1] *x[1]


    ave_pi = x[0] * piC + x[1] * piD + x[2] * piP

    function_1 = x[0] * (piC - ave_pi)

    function_2 = x[1] * (piD - ave_pi)

    function_3 = x[2] * (piP - ave_pi)

    function_4 = sit * x[3] * (1 - x[3]) * (piD - piC)

    function_5 = sit * x[-1] * (1 - x[-1]) * (piC - piD)

    return [function_1, function_2,function_3,function_4,function_5]


def plot_Time_evolution_chart(x,t):

    c = [sublist[0] for sublist in x]
    d = [sublist[1] for sublist in x]
    p = [sublist[2] for sublist in x]
    c_p = [sublist[3] for sublist in x]
    d_p = [sublist[4] for sublist in x]


    plt.plot(t,c)
    plt.xlabel('b')
    plt.ylabel('c')
    plt.title("Collaborator_ratio")
    plt.show()


    plt.plot(t,d)
    plt.xlabel('b')
    plt.ylabel('d')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(t,p)
    plt.xlabel('b')
    plt.ylabel('p')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(t,c_p)
    plt.xlabel('b')
    plt.ylabel('c_p')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(t,d_p)
    plt.xlabel('b')
    plt.ylabel('d_p')
    plt.title("Collaborator_ratio")
    plt.show()


def plot_variogram(x,b):

    c = [sublist[0] for sublist in x]
    d = [sublist[1] for sublist in x]
    p = [sublist[2] for sublist in x]
    c_p = [sublist[3] for sublist in x]
    d_p = [sublist[4] for sublist in x]


    plt.plot(b,c)
    plt.xlabel('b')
    plt.ylabel('c')
    plt.title("Collaborator_ratio")
    plt.show()


    plt.plot(b,d)
    plt.xlabel('b')
    plt.ylabel('d')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(b,p)
    plt.xlabel('b')
    plt.ylabel('p')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(b,c_p)
    plt.xlabel('b')
    plt.ylabel('c_p')
    plt.title("Collaborator_ratio")
    plt.show()

    plt.plot(b,d_p)
    plt.xlabel('b')
    plt.ylabel('d_p')
    plt.title("Collaborator_ratio")
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
    initial_x = [0.333, 0.333,0.333,0.5,0.5]
    t = list(range(1,10001))
    sit = 0.1

    # 1. 固定背叛诱惑b的时间演化图
    b = 2

    result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b,sit))
    plot_Time_evolution_chart(result,t)


   # 2. 背叛诱惑b变量的演化图
   #  result_finally=[]
   #  line_space_b=linespace(1,2,0.001)
   #  for b in tqdm(line_space_b):
   #      result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b,sit))
   #      result_finally.append(get_round(result[-1].tolist()))
   #  print(result_finally[-1])
   #  plot_variogram(result_finally, line_space_b)








