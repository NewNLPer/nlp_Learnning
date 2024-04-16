# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/6 18:52
coding with comment！！！
"""
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch

class Expert(nn.Module):
    # 简单起见，以MLP层作为展示
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)  # Gate layer
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        # 应用gate layer
        gate_scores = F.softmax(self.gate(x), dim=-1)  # s_len x num_experts
        # 获取每个token的top-2专家
        top2_experts = torch.topk(gate_scores, 2, dim=-1).indices  # s_len x 2
        # 初始化输出
        output = torch.zeros_like(x)
        # 处理每个专家
        for i in range(self.num_experts):
            # 筛选当前专家需要处理的tokens
            expert_mask = (top2_experts == i).any(dim=-1)  # s_len
            expert_input = x * expert_mask.unsqueeze(-1)  # s_len x dim
            # 计算当前专家的输出
            expert_output = self.experts[i](expert_input)
            # 加权平均
            weights = gate_scores[:, i].unsqueeze(-1)  # s_len x 1
            output += expert_output * weights
        return output

# 测试用例
batch_size = 2
seq_len = 10
input_dim = 5
output_dim = 3
num_experts = 4

# 随机生成输入数据
x = torch.randn(batch_size, seq_len, input_dim).cuda()


# 创建模型
moe = MixtureOfExperts(input_dim, output_dim, num_experts).cuda()

# 前向传播
output = moe(x)

# 检查输出维度
print(output.shape)

# 检查输出值
print(output)
