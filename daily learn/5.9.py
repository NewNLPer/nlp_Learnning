# -*- coding:utf-8 -*-
# @Time      :2022/5/9 9:56
# @Author    :Riemanner
# Write code with comments !!!
####回溯算法的总复习+总回顾
# def diStringMatch(s):
#     if len(s) == 1:
#         return [0]
#     else:
#         list1=list(range(len(s)+1))
#         list2=[]
#         start_zhen=0
#         end_zhen=len(list1)-1
#         for i in range(len(s)):
#             if s[i]=='D':
#                 list2.append(list1[end_zhen])
#                 end_zhen-=1
#             else:
#                 list2.append(list1[start_zhen])
#                 start_zhen+=1
#     return list2+list1[start_zhen:end_zhen+1]
# print(diStringMatch('DDI'))
# def reverseWords(s: str) -> str:
#     if len(s) == 1:
#         return s
#     else:
#         if s[0] != ' ':
#             s = ' ' + s
#         path = ''
#         k = 0
#         for i in range(len(s) - 1, -1, -1):
#             if k == 0 and s[i] == ' ':
#                 continue
#             elif s[i] == ' ':
#                 path = path + s[i:i + k + 1] + ''
#                 k = 0
#                 continue
#             else:
#                 k += 1
#         if path[0] == ' ':
#             path = path[1:]
#         if path[-1] == ' ':
#             path = path[:len(path) - 1]
#         return path
# print(reverseWords("the sky is blue"))
# def erfen(nums):
#     if len(nums)==1:
#         return nums[0]
#     else:
#         start_zhen=0
#         end_zhen=len(nums)-1
#         if nums[start_zhen]<nums[end_zhen]:
#             return nums[start_zhen]
#         else:
#             while start_zhen<end_zhen and end_zhen-start_zhen!=1:
#                 mid_zhen=(end_zhen+start_zhen)//2
#                 if nums[start_zhen]==nums[mid_zhen]==nums[end_zhen]:
#                     start_zhen+=1
#                     end_zhen-=1
#                     return erfen(nums[start_zhen:end_zhen+1])
#                 elif nums[start_zhen]<=nums[mid_zhen]:####说明左边有序，那我就不选择左边
#                     start_zhen=mid_zhen
#                 elif nums[mid_zhen]<=nums[end_zhen]:
#                     end_zhen=mid_zhen
#             return nums[end_zhen]
# print(erfen(nums = [2,2,2,0,1]))
# s=lambda x:x-1
# print(s(8))
class Node():
    def __init__(self,val):
        self.val=val
        self.next=None
a=Node(1)
b=Node(2)
lis=[]
lis.append(a)
print(lis)