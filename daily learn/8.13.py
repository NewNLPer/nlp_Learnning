# -*- coding:utf-8 -*-
# @Time      :2022/8/13 20:48
# @Author    :Riemanner
# Write code with comments !!!
def function(s1,s2):
    k=(s2[1]-s1[1])/(s2[0]-s1[0])
    b=s1[1]-k*s1[0]
    return (k,b)
def maxPoints(points):
    dic_c1={}
    dic_c2={}
    res = 0
    for each in points:  ###处理当k=inf时候的情况
        dic_c1[each[0]] = dic_c1.get(each[0], 0) + 1
        res = max(res,dic_c1[each[0]])
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            if points[i][0]!=points[j][0]:
                c=function(points[i],points[j])
                s1=set(dic_c2.get(c,{}))
                dic_c2[c]=s1.union({tuple(points[i]),tuple(points[j])})
                res=max(res,len(dic_c2[c]))
    return res


def largestLocal(grid):
    def funtion(i,j):
        return max(grid[i][j],
                   grid[i-1][j],
                   grid[i+1][j],
                   grid[i][j-1],
                   grid[i][j+1],
                   grid[i-1][j-1],
                   grid[i+1][j+1],
                   grid[i-1][j+1],
                   grid[i+1][j-1])
    m1=len(grid)###行
    grid1=[[0]*(m1-3+1) for _ in range(m1-3+1)]
    for i in range(m1-3+1):
        for j in range(m1-3+1):
            grid1[i][j]=funtion(i+1,j+1)
    return grid1

from functools import cache
def countSpecialNumbers(n) -> int:
    def bt(n,dic_c):
        if n<=10:
            return 10
        elif n in dic_c:
            return dic_c[n]
        else:
            if len(set(str(n)))==len(str(n)):
                dic_c[n]=bt(n-1,dic_c)+1
                return dic_c[n]
            else:
                dic_c[n]=bt(n-1,dic_c)
                return dic_c[n]
    return bt(n,{})


def nextPermutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    left, right = i + 1, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    return nums
def smallestNumber(pattern):
    res=list(range(1,len(pattern)+2))
    for i in range(len(res)):
        res[i]=str(res[i])
    while True:
        c=1
        for i in range(1,len(res)):
            if res[i]>res[i-1] and pattern[i-1]=='I':
                continue
            elif res[i]<res[i-1] and pattern[i-1]=='D':
                continue
            else:
                res=nextPermutation(res)
                c=0
                break
        if c==1:
            return ''.join(res)

import collections
def maxSlidingWindow(nums,k):
    res = []
    queue = collections.deque()
    for i, num in enumerate(nums):
        if queue and i-queue[0]==k:##说明维护的队列的第一个元素不在目前的滑动窗口内
            queue.popleft()
            print(1)
        while queue and nums[queue[-1]]<num:###单调递减
            queue.pop()
        queue.append(i)
        if i>=k-1:
            res.append(nums[queue[0]])
    return res

####二分法的记忆方法
###左闭右闭
def erfen1(nums,k):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<=end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]>k:
            end_zhen=mid-1
        elif nums[mid]<k:
            start_zhen=mid+1
        else:
            return mid
    return -1
def erfen2(nums,k):###左闭右开
    start_zhen=0
    end_zhen=len(nums)
    while start_zhen<end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]>k:
            end_zhen=mid
        elif nums[mid]<k:
            start_zhen=mid+1
        else:
            return mid
    return -1
