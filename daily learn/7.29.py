# -*- coding:utf-8 -*-
# @Time      :2022/7/29 9:06
# @Author    :Riemanner
# Write code with comments !!!
####前面第一个小于等于自己的元素
def fucntion1(nums:list):
    nums=[-pow(10,9)-1]+nums+[-pow(10,9)-1]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
####后面第一个小于自己的元素
def funtion2(nums:list):
    nums=[-pow(10,9)-1]+nums+[-pow(10,9)-1]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res

####前面第一个大于等于自己的元素
def fucntion3(nums:list):
    nums=[pow(10,9)+1]+nums+[pow(10,9)+1]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
####后面第一个大于自己的元素
def funtion4(nums:list):
    nums=[pow(10,9)+1]+nums+[pow(10,9)+1]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
def subArrayRanges(nums):
    res1=fucntion1(nums)
    res2=funtion2(nums)
    res3=fucntion3(nums)
    res4=funtion4(nums)
    res=0
    for i in range(1,len(nums)+1):
        res+=(nums[i-1]*(i-res3[i])*(res4[i]-i))
        res-=(nums[i-1]*(i-res1[i])*(res2[i]-i))
    return res
print(subArrayRanges(nums = [4,-2,-3,4,1]))
