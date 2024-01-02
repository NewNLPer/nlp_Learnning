import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import matplotlib

# 指定字体
matplotlib.rcParams['font.family'] = ['SimHei']  # 'SimHei' 是一种常用的中文黑体字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 参数设置
epsilon = 0.1
alpha = 0.1
beta = 0.01
theta = 0.1
gamma = 1  # 新增参数，调节氛围M的影响
b = 1.9    # 可以在[1, 2]范围内变化
mu = 0.1   # 一个示例值，可以调整

# 微分方程组
def model(y, t):
    x, M, H = y
    pi_C = x
    pi_D = b * x
    dxdt = x * (1 - x) * ( M * pi_C - (1 - M) * pi_D + (M - 0.5) ** 2)
    dMdt = epsilon * M * (1 - M) * (alpha * (x - (1 - x)) - beta * M**2 + mu * (H - (1 - H)))
    dHdt = theta * (x - H)
    return [dxdt, dMdt, dHdt]

# 初始条件
initial_condition = [0.5, 0.1, 0.1]

# 时间点
t = np.linspace(0, 10000, 10000)

# 求解微分方程组
solution = odeint(model, initial_condition, t)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(t, solution[:, 0], label='合作者比例 x')
plt.plot(t, solution[:, 1], label='氛围 M')
plt.plot(t, solution[:, 2], label='历史合作者比例 H')
plt.xlabel('时间')
plt.ylabel('比例')
plt.title('合作者和背叛者动态模型')
plt.legend()
plt.show()
