# -*- coding:utf-8 -*-
# @Time      :2022/7/9 10:37
# @Author    :Riemanner
# Write code with comments !!!

from typing import List

def maxSumDivThree(nums: List[int]) -> int:
    n = len(nums)
    dp = [[float("-inf") for _ in range(n)] for _ in range(3)]
    dp[0][0] = 0
    dp[nums[0] % 3][0] = nums[0]
    for i in range(1, n):
        r = nums[i] % 3
        if r == 0:
            dp[0][i] = max(dp[0][i - 1], dp[0][i - 1] + nums[i])
            dp[1][i] = max(dp[1][i - 1], dp[1][i - 1] + nums[i])
            dp[2][i] = max(dp[2][i - 1], dp[2][i - 1] + nums[i])
        elif r == 1:
            dp[0][i] = max(dp[0][i - 1], dp[2][i - 1] + nums[i])
            dp[1][i] = max(dp[1][i - 1], dp[0][i - 1] + nums[i])
            dp[2][i] = max(dp[2][i - 1], dp[1][i - 1] + nums[i])
        elif r == 2:
            dp[0][i] = max(dp[0][i - 1], dp[1][i - 1] + nums[i])
            dp[1][i] = max(dp[1][i - 1], dp[2][i - 1] + nums[i])
            dp[2][i] = max(dp[2][i - 1], dp[0][i - 1] + nums[i])
    return 0 if dp[0][-1] == float("-inf") else dp[0][-1]


def function(nums):
    n=len(nums)
    dp1=[0]*n
    dp1[0]=nums[0]
    dp2=[0]*n
    dp2[0]=nums[0]
    res=nums[0]
    for i in range(1,n):
        dp1[i]=max(dp1[i-1]+nums[i],nums[i])
        dp2[i]=min(dp2[i-1]+nums[i],nums[i])
        res=max(res,dp1[i],abs(dp2[i]))
    return res

def getDescentPeriods(prices: list) -> int:
    n=len(prices)
    dp=[1]*n###dp[i]表示price[i]为结尾的递减数目
    res=1
    for i in range(1,n):
        if prices[i]-prices[i-1]==-1:
            dp[i]=dp[i-1]+1
            res+=dp[i]
        else:
            res+=1
    return res


def numberOfWays(s):
    n = len(s)
    n1 = s.count('1')   # s 中 '1' 的个数
    res = 0   # 两种子序列的个数总和
    cnt = 0   # 遍历到目前为止 '1' 的个数
    for i in range(n):
        if s[i] == '1':
            res += (i - cnt) * (n - n1 - i + cnt)
            cnt += 1
        else:
            res += cnt * (n1 - cnt)
    return res
import random

class RandomizedSet:

    def __init__(self):
        self.s1={}
        self.s2=[]

    def insert(self, val: int) -> bool:
        if val not in self.s1:
            self.s1[val]=len(self.s2)
            self.s2.append(val)
            return True
        else:
            return False
    def remove(self, val: int) -> bool:
        if val in self.s1:
            index=self.s1[val]
            self.s2[index]=self.s2[-1]
            self.s2.pop()
            del self.s1[val]
            return True
        else:
            return False

    def getRandom(self) -> int:
        return random.choice(self.s2)

from collections import defaultdict

###dp[i][k]表示以nums[i]为结尾公差且大于等于2的等差数列的个数
def numberOfArithmeticSlices(A: List[int]) -> int:
    n = len(A)
    dp = [{} for _ in range(n)]
    res = 0
    for i, e in enumerate(A):
        for j, v in enumerate(A[:i]):
            diff = e - v
            dp[i][diff] = dp[i].get(diff, 0) + 1  #
            if diff in dp[j]:###这就说明一定大于等于3了
                # 以A[j]结尾,以diff为公差的等差数列个数.  可能存在重复,不止1个.
                dp[i][diff] += dp[j][diff]
                res += dp[j][diff]
    return res


def numRollsToTarget(n,k,target) -> int:
    dp=[[0]*(target+1) for _ in range(n+1)]
    dp[0][0]=1
    for i in range(1,target+1):
        if i<=k:
            dp[1][i]=1
    for i in range(1,n+1):
        for j in range(target,0,-1):
            if j-k>=0:
                start=j-k
            else:
                start=0
            dp[i][j]=sum(dp[i-1][start:j])%1000000007
    return dp[-1][-1]



# def numRollsToTarget(self, d: int, f: int, target: int) -> int:
#     if not d:
#         return 0
#     dp = [0] * (target + 1)
#     for i in range(1, target + 1):
#         if i <= f:
#             dp[i] = 1
#     for i in range(1, d):
#         for j in range(target, 0, -1):
#             start = j - f if j - f >= 0 else 0
#             dp[j] = sum(dp[start:j]) % 1000000007
#     return dp[target]

def findOcurrences(text: str, first: str, second: str):
    res=[]
    c=text.split(' ')
    start_zhen=0
    while start_zhen<len(c)-2:
        if c[start_zhen]==first and c[start_zhen+1]==second:
            res.append(c[start_zhen+2])
            start_zhen+=1
        else:
            start_zhen+=1
    return res

def getMaximumGenerated(n):
    if n==0:
        return 0
    else:
        dp=[0]*(n+1)
        dp[1]=1
        res=0
        i=1
        while i<len(dp)-1:
            if 2<=2*i<=n:
                dp[2*i]=dp[i]
                res=max(res,dp[2*i])
            if 2<=2*i+1<=n:
                dp[2*i+1]=dp[i]+dp[i+1]
                res=max(res,dp[2*i+1])
            i+=1
        return res
def maxTurbulenceSize(arr):
    n=len(arr)
    dp=[1]*n
    res=1
    for i in range(n-1):
        if i%2==0 and arr[i]<arr[i+1]:
            dp[i+1]=dp[i]+1
            res=max(res,dp[i+1])
        elif i%2==1 and arr[i]>arr[i+1]:
            dp[i+1]=dp[i]+1
            res=max(res,dp[i+1])
    dp2=[1]*n
    res1=1
    for i in range(n-1):
        if i%2==0 and arr[i]>arr[i+1]:
            dp2[i+1]=dp2[i]+1
            res1=max(res1,dp2[i+1])
        elif i%2==1 and arr[i]<arr[i+1]:
            dp2[i+1]=dp2[i]+1
            res1=max(res1,dp2[i+1])
    return max(res,res1)

def fillCups(amount):
    amount.sort()
    if amount[0]+amount[1]<=amount[2]:
        return amount[2]
    elif amount[0]==1 and amount[1]==amount[2]:
        return amount[1]+1
    else:
        if sum(amount)%2==0:
            return sum(amount)//2
        else:
            return sum(amount)//2+1

def goodDaysToRobBank(security,time):
    m = len(security)
    nincreasing = [0]*m
    ndecreasing = [0]*m
    for i in range(1, m):
        if security[i] <= security[i-1]:
            nincreasing[i] = nincreasing[i-1] + 1
        if security[-i-1] <= security[-i]:
            ndecreasing[-i-1] = ndecreasing[-i] + 1
    res = []
    for i in range(m):
        if nincreasing[i] >= time and ndecreasing[i] >= time:
            res.append(i)
    return res


def maxAlternatingSum(nums):
    n = len(nums)
    dp = [[0]*2 for _ in range(n)]  ###dp[i]表示0-i的最大交替和,二维表示长度
    dp[0][0]=0
    dp[0][1]=nums[0]
    for i in range(1,n):
        dp[i][0]=max(dp[i-1][1]-nums[i],dp[i-1][0])
        dp[i][1]=max(dp[i-1][1],dp[i-1][0]+nums[i])
    return max(dp[-1][0],dp[-1][1])


def lenLongestFibSubseq(arr):
    n=len(arr)
    dic_c={}
    for i in range(len(arr)):
        dic_c[arr[i]]=i
    dp=[[2]*n for _ in range(n)]
    res=0
    for i in range(1,n):
        for j in range(i+1,n):
            if arr[j]-arr[i] in dic_c and dic_c[arr[j]-arr[i]]<i:
                dp[i][j]=dp[dic_c[arr[j]-arr[i]]][i]+1
                res=max(res,dp[i][j])
    return res

def uniquePaths(m,n):
    dp=[[1]*n for _ in range(m)]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j]=dp[i-1][j]+dp[i][j-1]
    return dp[-1][-1]
from functools import lru_cache
def longestIncreasingPath(matrix):
    n=len(matrix)
    m=len(matrix[0])
    cc=0
    @lru_cache()
    def f_t(i,j):
        s=[(1,0),(0,1),(-1,0),(0,-1)]
        res=1
        for x,y in s:
            if 0<=i+x<n and 0<=j+y<m:
                if matrix[i+x][j+y]>matrix[i][j]:
                    res=max(res,1+f_t(i+x,j+y))
        return res
    for i in range(n):
        for j in range(m):
            cc=max(cc,f_t(i,j))
    return cc


def minPathSum(grid):
    n=len(grid)
    m=len(grid[0])
    dp=[[0]*m for _ in range(n)]
    dp[0][0]=grid[0][0]
    for i in range(1,m):
        dp[0][i]=dp[0][i-1]+grid[0][i]
    for j in range(1,n):
        dp[j][0]=dp[j-1][0]+grid[j][0]
    for i in range(1,n):
        for j in range(1,m):
            dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j]
    return dp[-1][-1]


def minCut(s):
###现进行字符串的判断，然后在确定分割次数dp[i][j] i-j是不是回文字符串，包含i也包含j
    n=len(s)
    dp1=[[0]*n for _ in range(n)]
    for i in range(n-1,-1,-1):
        for j in range(i,n):
            if s[i]!=s[j]:
                dp1[i][j]=0
            elif j-i<=1 or (s[i]==s[j] and dp1[i+1][j-1]):
                dp1[i][j]=1
    dp2=[9999999]*n
    dp2[0]=0
    for i in range(1,n):
        if dp1[0][i]:
            dp2[i]=0
        else:
            for j in range(i):
                if dp1[j+1][i]:
                    dp2[i]=min(dp2[i],dp2[j]+1)
    return dp2[-1]


def getKthMagicNumber(k):
    res=[1]
    res1=[0,0,0]
    while len(res)<k:
        c=min(res[res1[0]]*3,res[res1[1]]*5,res[res1[2]]*7)
        if res[-1]!=c:
            res.append(c)
        if res[-1]==res[res1[0]]*3:
            res1[0]+=1
        elif res[-1]==res[res1[1]]*5:
            res1[1]+=1
        elif res[-1]==res[res1[2]]*7:
            res1[2]+=1
    return res
