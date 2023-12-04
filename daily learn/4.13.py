####策略的生成
# import pandas as pd
# import numpy as np
# csv_path = 'ceshi.csv'
# o_data = pd.read_csv(csv_path)
# o_data.dropna(how='all', inplace=True)
# data = np.array(o_data)
# def product(s, repeat=1):
#     pools = [tuple(pool) for pool in s] * repeat
#     result = [[]]
#     for pool in pools:
#         result = [x+[y] for x in result for y in pool]
#     for prod in result:
#         yield tuple(prod)
# def Initial_formation_strategy(data):
#     def shengcheng(data):
#         New_strategy = [[]] * 5
#         for i in range(5):
#             for j in range(len(data)-6, len(data)):
#                 if data[j][i] == 0:
#                     if len(data) - 1 - j <= 2:
#                         New_strategy[i] = [0]
#                     if len(data) - 1 - j > 2:
#                         New_strategy[i] = [0.5, 1]
#                 if data[j][i]==1 and j==len(data)-1:
#                     New_strategy[i] = [0,0.5, 1]
#                 if data[j][i]==1 and j!=len(data)-1:
#                     New_strategy[i] = [1]
#                 else:
#                     New_strategy[i]=[0,0.5,1]
#         print(New_strategy)
#         return New_strategy
#     result=product(shengcheng(data))
#     data_use=[]
#     for i in result:
#         i=list(i)
#         data_use.append(i)
#     return np.array(data_use)
# data_use1=Initial_formation_strategy(data)
# print(data_use1)

#####进行回溯算法的学习
#
# def combinationArray(self, candidates, target):
#     candidates.sort()
#     res = []  # 存放组合结果
#     size = len(candidates)
#
#     def backtrack(combination, cur_sum, j):
#         # combination目前已经产生的组合,cur_sum当前计算和，j用于控制求和的查找范围起点
        # 递归出口
#         if cur_sum > target:
#             return
#         if cur_sum == target:
#             res.append(combination)
#         for i in range(j, size):  # j避免重复
#             if cur_sum + candidates[i] > target:  # 约束函数(剪)
#                 break
#             j = i
#             backtrack(combination + [candidates[i]], cur_sum + candidates[i], j)  # 递归回溯
#
# #     backtrack([], 0, 0)
#
#             def combinationArray(self, candidates, target):
#
#                 candidates.sort()
#                 res = []  # 存放组合结果
#                 size = len(candidates)
#
#                 def backtrack(combination, cur_sum, j):
#                     # combination目前已经产生的组合,cur_sum当前计算和，j用于控制求和的查找范围起点
#                     # 递归出口
#                     if cur_sum > target:
#                         return
#                     if cur_sum == target:
#                         res.append(combination)
#                     for i in range(j, size):  # j避免重复
#                         if cur_sum + candidates[i] > target:  # 约束函数(剪)
#                             break
#                         j = i
#                         backtrack(combination + [candidates[i]], cur_sum + candidates[i], j)  # 递归回溯
#
#                 backtrack([], 0, 0)
#                 return res
#
#         if __name__ == '__main__':
#             candidates = [2, 3, 6, 7]
#             target = 7
#             solution = Solution()
#             print(solution.combinationArray(candidates, target))
#         class Solution:
#             def combinationArray(self, candidates, target):
#
#                 candidates.sort()
#                 res = []  # 存放组合结果
#                 size = len(candidates)
#
#                 def backtrack(combination, cur_sum, j):
#                     # combination目前已经产生的组合,cur_sum当前计算和，j用于控制求和的查找范围起点
#                     # 递归出口
#                     if cur_sum > target:
#                         return
#                     if cur_sum == target:
#                         res.append(combination)
#                     for i in range(j, size):  # j避免重复
#                         if cur_sum + candidates[i] > target:  # 约束函数(剪)
#                             break
#                         j = i
#                         backtrack(combination + [candidates[i]], cur_sum + candidates[i], j)  # 递归回溯
#
#                 backtrack([], 0, 0)
#                 return res
#
#         if __name__ == '__main__':
#             candidates = [2, 3, 6, 7]
#             target = 7
#             solution = Solution()
# #             print(solution.combinationArray(candidates, target))
# def combinationArray(candidates, target):
#     candidates.sort()
#     res = []  # 存放组合结果
#     size = len(candidates)
#
#     def backtrack(combination, cur_sum, j):
#         # combination目前已经产生的组合,cur_sum当前计算和，j用于控制求和的查找范围起点
#         # 递归出口
#         if cur_sum > target:
#             return
#         if cur_sum == target:
#             res.append(combination)
#         for i in range(j, size):  # j避免重复
#             if cur_sum + candidates[i] > target:  # 约束函数(剪)
#                 break
#             j = i
#             backtrack(combination + [candidates[i]], cur_sum + candidates[i], j)  # 递归回溯
#
#     backtrack([], 0, 0)
# def combinationSum(candidates, target):
#     candidates.sort()
#     n = len(candidates)
#     res = []
#     def helper(i, tmp_sum, tmp):
#         if tmp_sum > target or i == n:
#             return
#         if tmp_sum == target:
#             res.append(tmp)
#             return
#         helper(i, tmp_sum + candidates[i], tmp + [candidates[i]])
#         helper(i + 1, tmp_sum, tmp)
#     helper(0, 0, [])
#     return res
# print(combinationSum([2,3,5],8))
import copy
def combinationSum(candidates, target):
    def bt(candidates, path, target):
        if sum(path) == target:  # path和等于target时，加入结果集
            res.append(copy.deepcopy(path))
            return
        elif sum(path) > target:  # path和大于target时，剪枝
            return
        for i in range(0, len(candidates)):  # 遍历选择列表
            path.append(candidates[i])  # 做出选择
            bt(candidates[i:], path, target)  # 进入下一层，注意candidates[i:]保证同一层不同广度的选择列表里没有已经选择过的数
            path.pop()  # 撤销选择

    res = []  # 结果集
    bt(candidates, [], target)  # 开始递归
    return res  # 返回结果集
print(combinationSum([2,3,5],8))
