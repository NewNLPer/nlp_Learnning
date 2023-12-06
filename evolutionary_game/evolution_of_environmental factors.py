# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/6 14:52
coding with comment！！！
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def Cooperation_proportion_derivatives(x, t, punish, b, xi):
    function_1 = x[0] * (1 - x[0]) * (x[0] - b + punish * (x[1] * x[0] - 2 * x[1] + 1))
    function_2 = xi * x[1] * (1 - x[1]) * ((1 - x[0]) * b - x[0] * x[0] + punish * (x[0] - 1 + x[1] - x[1] * x[0] * x[0]))
    return [function_1, function_2]

def get_plot(x,t):

    Collaborator_ratio= [sublist[0] for sublist in x]

    Perceived_probability=[sublist[1] for sublist in x]

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(t,Collaborator_ratio)
    plt.xlabel('t')
    plt.ylabel('c')
    plt.title("Collaborator_ratio")

    plt.subplot(2, 1, 2)
    plt.plot(t,Perceived_probability)
    plt.xlabel('t')
    plt.ylabel('p')
    plt.title("Perceived_probability")

    plt.suptitle("Copy equation simulation results")
    plt.show()



if __name__=="__main__":

    initial_x=[0.5, 0.5] ##系统初始条件
    t = np.linspace(0, 800, 800)  #时间戳
    punish = 0.5
    b = 1.5
    xi = 0.01
    result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(punish, b, xi))
    get_plot(result,t)


