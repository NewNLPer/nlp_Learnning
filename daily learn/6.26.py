# -*- coding:utf-8 -*-
# @Time      :2022/6/26 8:39
# @Author    :Riemanner
# Write code with comments !!!
import numpy
class Solution:

    def __init__(self, n,blacklist):
        self.k=set(blacklist)
        self.j=set(range(n))-self.k
    def pick(self) -> int:
        p = numpy.array([1/len(self.j)]*len(self.j))
        index = numpy.random.choice(list(self.j), p=p.ravel())
        return int(index)
A=Solution(7,[2,3,5])

def lengthOfLongestSubstring(s):
    dic_c={}
    start_zhen=0
    end_zhen=0
    res=0
    while end_zhen<len(s):
        if s[end_zhen] not in dic_c or dic_c[s[end_zhen]]==0:
            dic_c[s[end_zhen]]=1
            end_zhen+=1
            res=max(res,sum(dic_c.values()))
        else:
            while dic_c[s[end_zhen]]!=0:
                dic_c[s[start_zhen]]-=1
                start_zhen+=1
            dic_c[s[end_zhen]]=1
            end_zhen+=1
    return res

def nthUglyNumber(n):
    res=[1]
    table=[0,0,0]
    s=1
    while s<n:
        min1=min([res[table[0]]*2,res[table[1]]*3,res[table[2]]*5])
        if min1!=res[-1]:
            res.append(min1)
        else:
            s-=1
        if min1==res[table[0]]*2:
            table[0]+=1
            s+=1
        elif min1==res[table[1]]*3:
            table[1]+=1
            s+=1
        elif min1==res[table[2]]*5:
            table[2]+=1
            s+=1
    return res[-1]

def constructArr(a):
    s1=[1]*len(a)
    s2=[1]*len(a)
    for i in range(1,len(a)):
        s1[i]=s1[i-1]*a[i-1]
    for j in range(len(a)-2,-1,-1):
        s2[j]=s2[j+1]*a[j+1]
    s3=[1]*len(a)
    for k in range(len(a)):
        s3[k]=s1[k]*s2[k]
    return s3




def maxProfit(prices):
    n=len(prices)
    if n==0:
        return 0
    else:
        dp=[[0]*5 for _ in range(n)]
        dp[0][0]=0
        dp[0][1]=-prices[0]
        dp[0][2]=0
        dp[0][3]=-prices[0]
        dp[0][4]=0
        for i in range(1,n):
            dp[i][0]=dp[i-1][0]
            dp[i][1]=max(dp[i-1][0]-prices[i],dp[i-1][1])
            dp[i][2]=max(dp[i-1][1]+prices[i],dp[i-1][2])
            dp[i][3]=max(dp[i-1][2]-prices[i],dp[i-1][3])
            dp[i][4]=max(dp[i-1][3]+prices[i],dp[i-1][4])
        print(dp)
        return dp[-1][-1]

def maxProfit1(k,prices):
    n=len(prices)
    if n==0:
        return 0
    else:
        dp=[[0]*(2*k+1) for _ in range(n)]
        dp[0][0]=0
        for j in range(1,2*k+1):
            if j%2!=0:
                dp[0][j]=-prices[0]
            else:
                dp[0][j]=0
        for i in range(1,n):
            for j in range(0,2*k-1,2):
                dp[i][j+1]=max(dp[i-1][j+1],dp[i-1][j]-prices[i])
                dp[i][j+2]=max(dp[i-1][j+2],dp[i-1][j+1]+prices[i])
        print(dp)
        return dp[-1][-1]




def isStraight(nums):
    res=[]
    res1=0
    dic_c={'J':11,'Q':12,'K':13,'A':1}
    for num in nums:
        if num in dic_c:
            res.append(dic_c[num])
        elif num==0:
            res1+=1
        else:
            res.append(num)
    res.sort()
    res2=0
    for i in range(1,len(res)):
        path=res[i]-res[i-1]-1
        if path<0:
            return False
        elif path>0:
            res2+=path
    return res1>=res2

def search(nums,target):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<=end_zhen:
        c=0
        mid=(start_zhen+end_zhen)//2
        if nums[mid]==target:
            c=1
            break
        elif nums[mid]<target:
            start_zhen=mid+1
        else:
            end_zhen=mid-1
    if c==0:
        return 0
    else:
        res=1
        qian=mid+1
        hou=mid-1
        while qian<len(nums):
            if nums[qian]==target:
                res+=1
                qian+=1
            else:
                break
        while hou>=0:
            if nums[hou]==target:
                res+=1
                hou-=1
            else:
                break
        return res

import collections
def maxSlidingWindow(nums,k):
    res = []
    queue = collections.deque()
    for i, num in enumerate(nums):
        if queue and i-queue[0]==k:##说明维护的队列的第一个元素不在目前的滑动窗口内
            queue.popleft()
        while queue and nums[queue[-1]]<num:
            queue.pop()
        queue.append(i)
        if i>=k-1:
            res.append(nums[queue[0]])
    return res
print(maxSlidingWindow(nums = [3,1,-1,-3,5,3,6,7], k = 3))