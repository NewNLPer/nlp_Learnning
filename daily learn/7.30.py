# -*- coding:utf-8 -*-
# @Time      :2022/7/30 16:58
# @Author    :Riemanner
# Write code with comments !!!
####前面第一个小于自己的元素
def fucntion1(nums:list):
    nums=[-pow(10,9)-1]+nums+[-pow(10,9)-1]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>=nums[i]:
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


def longestWPI(hours):
    n = len(hours)
    # 大于8小时计1分 小于等于8小时计-1分
    score = [0] * n
    for i in range(n):
        if hours[i] > 8:
            score[i] = 1
        else:
            score[i] = -1
    # 前缀和
    presum = [0] * (n + 1)
    for i in range(1, n + 1):
        presum[i] = presum[i - 1] + score[i - 1]
    ans = 0
    stack = []
    # 顺序生成单调栈，栈中元素从第一个元素开始严格单调递减，最后一个元素肯定是数组中的最小元素所在位置
    for i in range(n + 1):
        if not stack or presum[stack[-1]] > presum[i]:
            stack.append(i)
    # 倒序扫描数组，求最大长度坡
    i = n
    while i > ans:
        while stack and presum[stack[-1]] < presum[i]:
            ans = max(ans, i - stack[-1])
            stack.pop()
        i -= 1
    return ans
def maxWidthRamp(nums: list[int]) -> int:
    stack=[]
    res1=[0]*len(nums)
    for i in range(len(nums)):
        if not stack or nums[i]<nums[stack[-1]]:
            stack.append(i)
    for i in range(len(nums)-1,-1,-1):
        for j in range(len(stack)-1,-1,-1):
            if nums[i]>nums[stack[j]]:
                res1[i]=max(res1[i],i-stack[j])
    return res1


print(maxWidthRamp([6,0,8,2,1,5]))