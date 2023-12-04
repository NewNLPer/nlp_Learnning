####动态规划写完
import numpy as np
# m_list=[(1,2,3),(3,4,5)]
# m_arr=np.array(m_list) # 转为数组
# m_arr=np.append(m_arr,[[1,1,1]],axis=0) # 添加整行元素
# print(m_arr)
# # print(m_arr)
# # np.append(m_arr,[[1],[1]],axis=1) # 添加整列
# # print(m_arr)
#
# data = np.random.randint(low=22,high=30,size=(len(data),1))
# def combinationSum(candidates, target):
#     if len(candidates) == 1:
#         if target % candidates[0] == 0:
#             return [[candidates[0]] * (target // candidates[0])]
#         else:
#             return []
#     else:
#         if candidates[0] > target:
#             return []
#         else:
#             S = []
#             for i in range(len(candidates) - 1, -1, -1):
#                 if target % candidates[i] == 0:
#                     S.append([candidates[i]]* (target // candidates[i]))
#                 elif target % candidates[i] != 0 and (target - candidates[i]) < candidates[0]:
#                     continue
#                 elif target % candidates[i] != 0 and (target - candidates[i]) >= candidates[0] and combinationSum(candidates, target - candidates[i]) != []:
#                     S.append([candidates[i]]+combinationSum(candidates, target - candidates[i]))
#                 else:
#                     continue
#         return S
#
# print(combinationSum([2,3,5],8))

d=np.zeros(5)
print(d)
print(d.T)
