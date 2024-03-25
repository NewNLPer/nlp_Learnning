# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/2/25 15:05
coding with comment！！！
"""


import matplotlib.pyplot as plt
from scipy.integrate import odeint


initial_x = [0.2, 0.3]
t = list(range(1, 200001))
alph = 0.2
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


def time_evloution(b,k_1,k_2):
    if k_1 == -1: # k_1 = [0,0.3,0.7,1]
        for k_1 in [0,0.3,0.7,1]:
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit,k_1,k_2))
            Collaborator_ratio = [sublist[0] for sublist in result]
            plt.xlabel('t')
            plt.ylabel(r'$\rho_c$')
            plt.semilogx(t, Collaborator_ratio, label="$k_{1}=%s$"%(k_1))
        plt.legend()
        plt.title(r'$b=1.5$, $k_{2}=0.5$, $\varepsilon=%s$'%sit)
        plt.show()

        for k_1 in [0,0.3,0.7,1]:
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit,k_1,k_2))
            Degree_of_rewards_and_punishments = [sublist[1] for sublist in result]
            plt.xlabel('t')
            plt.ylabel('M')
            plt.semilogx(t,Degree_of_rewards_and_punishments,label="$k_{1}=%s$"%(k_1))
        plt.legend()
        plt.title(r'$b=1.5$, $k_{2}=0.5$, $\varepsilon=%s$'%sit)
        plt.show()

    if k_2 == -1: # k_1 = [0,0.3,0.7,1]
        for k_2 in [0,0.3,0.7,1]:
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit,k_1,k_2))
            Collaborator_ratio = [sublist[0] for sublist in result]
            plt.xlabel('t')
            plt.ylabel(r'$\rho_c$')
            plt.semilogx(t,Collaborator_ratio,label="$k_{2}=%s$"%(k_2))
        plt.legend()
        plt.title('$b=1.5,k_{1}=0.5$')
        plt.show()

        for k_2 in [0,0.3,0.7,1]:
            result = odeint(Cooperation_proportion_derivatives, initial_x, t, args=(b, alph, yit, sit,k_1,k_2))
            Degree_of_rewards_and_punishments = [sublist[1] for sublist in result]
            plt.xlabel('t')
            plt.ylabel('M')
            plt.semilogx(t,Degree_of_rewards_and_punishments,label="$k_{2}=%s$"%(k_2))
        plt.legend()
        plt.title('$b=1.5,k_{1}=0.5$')
        plt.show()


if __name__=="__main__":

    time_evloution(1.5,-1,0.5)
