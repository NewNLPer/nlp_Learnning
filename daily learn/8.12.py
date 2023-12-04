# -*- coding:utf-8 -*-
# @Time      :2022/8/12 16:46
# @Author    :Riemanner
# Write code with comments !!!
###找前面第一个比我小的
def funtion1(nums):
    stack=[]
    res=[0]*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
###找后面第一个比我小的
def funtion2(nums):
    stack=[]
    res=[0]*len(nums)
