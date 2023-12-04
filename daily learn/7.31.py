# -*- coding:utf-8 -*-
# @Time      :2022/7/31 8:49
# @Author    :Riemanner
# Write code with comments !!!
####前面第一个小于于自己的元素
def fucntion1(nums:list):
    nums=[0]+nums+[0]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
####后面第一个小于自己的元素
def funtion2(nums:list):
    nums=[0]+nums+[0]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res


