# -*- coding:utf-8 -*-
# @Time      :2022/6/30 8:33
# @Author    :Riemanner
# Write code with comments !!!
from functools import lru_cache
def is_prime3(x):
    if x==1:
        return False
    elif x == 2:
        return True
    elif x % 2 == 0:
        return False
    for i in range(3, int(x ** 0.5) + 1, 2):
        if x % i == 0:
            return False
    return True
def numPrimeArrangements(n):
    if n==1:
        return 0
    elif n==2:
        return 1
    elif n==3:
        return 2
    else:
        @lru_cache()
        def jiechegn(n):
            if n==1:
                return 1
            else:
                return n*jiechegn(n-1)
        res=0
        for i in range(1,n+1):
            if is_prime3(i):
                res+=1
        return (jiechegn(res)*jiechegn(n-res))%(pow(10,9)+7)


def zixu_lie(s1,s2):#判断s1是否是s2的子序列
    start_zhen1=0
    start_zhen2=0
    while start_zhen1<len(s1) and start_zhen2<len(s2):
        if s1[start_zhen1]==s2[start_zhen2]:
            start_zhen1+=1
            start_zhen2+=1
        else:
            start_zhen2+=1
    if start_zhen1==len(s1):
        return True
    else:
        return False


def findLUSlength(strs: list[str]) -> int:
    res=-1
    for i in range(len(strs)):
        c=1
        for j in range(len(strs)):
            if i==j:
                continue
            elif len(strs[i])>len(strs[j]) or not zixu_lie(strs[i],strs[j]):
                continue
            else:
                c=0
                break
        if c==1:
            res=max(res,len(strs[i]))
    return res

def countSubstrings(s: str) -> int:
    n1=len(s)
    res=0
    dp=[[0]*n1 for _ in range(n1)]
    for i in range(n1):
        dp[i][i]=1
        res+=1
    for i in range(n1-1,-1,-1):
        for j in range(i,n1):
            if i==j:
                continue
            elif s[i]==s[j]:
                if j-i==1:
                    dp[i][j]=1
                    res+=1
                else:
                    dp[i][j]=dp[i+1][j-1]
                    if dp[i][j]==1:
                        res+=1
    return res

import sys
def minCut(s):
    isPalindromic = [[False] * len(s) for _ in range(len(s))]
    for i in range(len(s) - 1, -1, -1):
        for j in range(i, len(s)):
            if s[i] != s[j]:
                isPalindromic[i][j] = False
            elif j - i <= 1 or isPalindromic[i + 1][j - 1]:
                isPalindromic[i][j] = True
    dp = [sys.maxsize] * len(s)
    dp[0] = 0
    for i in range(1, len(s)):
        if isPalindromic[0][i]:
            dp[i] = 0
            continue
        for j in range(0, i):
            if isPalindromic[j + 1][i] == True:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[-1]


def maxCoins(nums):
    #nums首尾添加1，方便处理边界情况
    # dp[i][j] 表示开区间 (i,j) 内你能拿到的最多金币
    nums.insert(0,1)
    nums.insert(len(nums),1)
    dp=[[0]*len(nums) for _ in range(len(nums))]
    res=0
    for i in range(len(nums)):
        for j in range(i,len(nums)):
            if j-i==2:
                dp[i][j]=nums[i]*nums[i+1]*nums[i+2]
    for i in range(len(nums)-1,-1,-1):
        for j in range(i+1,len(nums)):
            for k in range(i+1,j):
                dp[i][j]=max(dp[i][k]+dp[k][j]+nums[i]*nums[k]*nums[j],dp[i][j])
                res=max(res,dp[i][j])
    return res

def longestIncreasingPath(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    lst = []
    for i in range(m):
        for j in range(n):
            lst.append((matrix[i][j], i, j))
    lst.sort()
    print(lst)
    dp = [[0 for _ in range(n)] for _ in range(m)]
    for num, i, j in lst:
        dp[i][j] = 1
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = i + di, j + dj
            if 0 <= r < m and 0 <= c < n:
                if matrix[i][j] > matrix[r][c]:
                    dp[i][j] = max(dp[i][j], 1 + dp[r][c])
    return max([dp[i][j] for i in range(m) for j in range(n)])


def eraseOverlapIntervals(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    n = len(intervals)
    right = intervals[0][1]
    ans = 1
    for i in range(1, n):
        if intervals[i][0] >= right:
            ans += 1
            right = intervals[i][1]
    return n - ans


def updateMatrix(mat):
    n=len(mat)
    m=len(mat[0])
    dp=[[0]*m for _ in range(n)]
    dic_c=[(0,1),(1,0),(-1,0),(0,-1)]
    for i in range(n):
        for j in range(m):
            if mat[i][j]==0:
                dp[i][j]=0
                continue
            else:
                res=1
                for each in dic_c:
                    if i+each[0]>=0 and i+each[0]<n and j+each[1]>=0 and j+each[1]<m:
                        res=min(res,mat[i+each[0]][j+each[1]])
                if res==0:
                    dp[i][j]=1
                else:
                    dp[i][j]='#'
    return dp

def findNumberOfLIS(nums):
    n=len(nums)
    dp=[1]*n
    dp[0]=1
    dic_c={}
    res=0
    for i in range(1,n):
        for j in range(i):
            if nums[i]>nums[j]:
                dp[i]=max(dp[i],dp[j]+1)
                res=max(res,dp[i])
        dic_c[dp[i]]=dic_c.get(dp[i],0)+1
    if dic_c[res]>1:
        return dic_c[res]
    elif dic_c[res]==1:
        return dic_c[res-1]
print(findNumberOfLIS([2,2,2,2,2]))