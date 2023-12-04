# -*- coding:utf-8 -*-
# @Time      :2022/7/2 10:11
# @Author    :Riemanner
# Write code with comments !!!


import heapq


def minRefuelStops(target: int, startFuel: int, stations) -> int:
    if target <= startFuel: return 0
    heap = []
    remainOil = startFuel  # 剩余的汽油
    pos = 0  # 经过的加油站
    res = 0  # 加油次数
    while remainOil < target:  # 没油了
        for i in range(pos, len(stations)):
            if remainOil >= stations[i][0]:  # 可以到达这个加油站
                heapq.heappush(heap, -stations[i][1])  # 带上这桶油
                pos += 1  # 这个加油站已经路过了
            else:
                break
        if remainOil < target:#还没到中点
            if not heap:  # 身上没油了
                return -1
            remainOil -= heapq.heappop(heap)  # python 只有最小堆 这里是取负数，选择最大的加油
            res += 1  # 加油次数加一
    return res

def deleteAndEarn(nums):
    dic_c={}
    res=[0]*(max(nums)+1)
    for i in nums:
        dic_c[i]=dic_c.get(i,0)+1
    for key in dic_c:
        res[key]=key*dic_c[key]
    dp=[0]*len(res)
    dp[0]=res[0]
    dp[1]=max(res[0],res[1])
    for i in range(2,len(res)):
        dp[i]=max(dp[i-2]+res[i],dp[i-1])
    return dp[-1]

### dp[i][j]表示先出手的玩家在i-j区间内可以或得到的最大收益
def canIWin(maxChoosableInteger,desiredTotal):
    dp=[[0]*(maxChoosableInteger+1) for _ in range(maxChoosableInteger+1)]
    for i in range(maxChoosableInteger+1):
        dp[i][i]=i
    for i in range(1,maxChoosableInteger+1):
        dp[i-1][i]=i
    for i in range(maxChoosableInteger-1,-1,-1):
        for j in range(i+2,maxChoosableInteger+1):
            dp[i][j]=max(i+min(dp[i+1][j-1],dp[i+2][j]),j+min(dp[i][j-2],dp[i+1][j-1]))
    print(dp)
    return dp[1][-1]>=desiredTotal


def findPaths(m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    ans,mod = 0,10**9 + 7
    dp = [[[0]*(maxMove+1) for _ in range(n)] for _ in range(m)]
    dp[startRow][startColumn][0]=1
    for k in range(maxMove):
        print(dp)
        for i in range(m):
            for j in range(n):
                if dp[i][j][k]>0:
                    for ii,jj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                        if 0<= ii <m and 0<= jj <n:
                            dp[ii][jj][k+1] = (dp[ii][jj][k+1] + dp[i][j][k]) % mod
                        else:
                            ans = (ans+dp[i][j][k]) % mod
    return ans

'''
状态定义：dp[i][j][k]表示球k次操作后移到i,j位置的路径数
状态转移：遍历K次矩阵m*n 且 dp[i][j][k]>0有值的情况下，尝试相邻四个方向作为k+1次操作
若k+1次未出界，则把第k次dp[i][j][k]累加到dp[i][j][k+1]
若k+1次出界，则把第k次dp[i][j][k]累加到最终结果
边界情况：dp[startRow][startColumn][0]=1
'''






def decodeMessage(key: str, message: str) -> str:
    dic_c1=set()
    dic_c2={}
    stat_zhen=0
    s=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i in key:
        if i==' ':
            continue
        else:
            if i not in dic_c1:
                dic_c2[i]=s[stat_zhen]
                stat_zhen+=1
                dic_c1.add(i)
    s=''
    for i in message:
        if i==' ':
            s+=' '
        else:
            s+=dic_c2[i]
    return s

def longestIncreasingPath(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    lst = []
    res=m*n
    for i in range(m):
        for j in range(n):
            lst.append((matrix[i][j], i, j))
    lst.sort()
    dp = [[0 for _ in range(n)] for _ in range(m)]
    for num, i, j in lst:
        dp[i][j] = 1
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = i + di, j + dj
            if 0 <= r < m and 0 <= c < n:
                if matrix[i][j] > matrix[r][c]:
                    dp[i][j] = max(dp[i][j], 1 + dp[r][c])
                    res+=1
    if m==1 or n==1:
        return res
    else:
        return res+1

def findLongestChain(pairs:list) -> int:
    n=len(pairs)
    dp=[0]*n
    res=1
    pairs.sort(key=lambda x:x[1])
    dp[0]=1
    for i in range(1,n):
        for j in range(i):
            if pairs[i][0]>pairs[j][1]:
                dp[i]=max(dp[j]+1,dp[i])
                res=max(res,dp[i])
    return res

def minSteps(n: int) -> int:
    dp=[n]*(n+1)
    dp[1]=0
    for i in range(2,n+1):
        for j in range(1,i):
            if i%j==0:
                dp[i]=min(dp[j]+(i//j),dp[i])
    return dp[-1]

def minimumDeleteSum(s1: str, s2: str) -> int:
    n=len(s1)+1
    m=len(s2)+1
    dp=[[0]*m for _ in range(n)]
    for i in range(1,n):
        dp[i][0]=(dp[i-1][0]+ord(s1[i-1]))
    for j in range(1,m):
        dp[0][j]=(dp[0][j-1]+ord(s2[j-1]))
    for i in range(1,n):
        for j in range(1,m):
            if s1[i-1]==s2[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(dp[i-1][j]+ord(s1[i-1]),dp[i][j-1]+ord(s2[j-1]),dp[i-1][j-1]+ord(s1[i-1])+ord(s2[j-1]))
    return dp[-1][-1]

def checkValidString(s: str) -> bool:
    q, star = [], []
    for i, c in enumerate(s):
        if c == '(': q.append(i)
        if c == '*': star.append(i)
        if c == ')':
            if q:
                q.pop()
            else:
                if star:
                    star.pop()
                else:
                    return False
    while q:
        i = q.pop()
        if star:
            j = star.pop()
            if i > j:
                return False
        else:
            return False
    return True

def findLength(nums1,nums2) -> int:
    n=len(nums1)+1
    m=len(nums2)+1
    dp=[[0]*m for _ in range(n)]
    c1=0
    c2=0
    for i in range(1,n):
        for j in range(1,m):
            if i==1 or j==1:
                if j==1:
                    if nums1[i - 1] == nums2[0]:
                        dp[i][1] = 1
                        c1 = 1
                    else:
                        if c1 == 1:
                            dp[i][1] = 1
                elif i==1:
                    for j in range(1, m):
                        if nums2[j - 1] == nums1[0]:
                            dp[1][j] = 1
                            c2 = 1
                        else:
                            if c2 == 1:
                                dp[1][j] = 1
            elif nums1[i-1]==nums2[j-1]:
                cum=dp[i-1][j-1]
                if nums1[i-cum-1:i-1]==nums2[j-cum-1:j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                else:
                    dp[i][j]=max(1,dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[-1][-1]

def minimumAbsDifference(arr:list):
    arr.sort()
    dic_c={}
    res=1000000
    for i in range(1,len(arr)):
        dic_c[abs(arr[i]-arr[i-1])]=dic_c.get(abs(arr[i]-arr[i-1]),[])+[[arr[i-1],arr[i]]]
        res=min(res,abs(arr[i]-arr[i-1]))
    return dic_c[res]

def rotatedDigits(n: int) -> int:
    dp=[0]*(n+1)
    dp[1]=0
    for i in range(2,n+1):
        s=str(i)
        if '3' in s or '4' in s or '7' in s:
            dp[i]=dp[i-1]
        elif ({'0','1','8'}|set(s))=={'0','1','8'}:
            dp[i]=dp[i-1]
        else:
            dp[i]=dp[i-1]+1


def findContentChildren(g,s):
    s.sort()###饼干
    g.sort()###孩子的胃口值
    end_zhen=len(s)-1
    res=0
    for i in range(len(g)-1,-1,-1):
        if end_zhen<0:
            return res
        elif g[i]<=s[end_zhen]:
            res+=1
            end_zhen-=1
    return res


def jump(nums):
    if len(nums) == 1:
        return 0
    elif len(nums) == 2:
        return 1
    else:
        if nums[0]+1 >= len(nums):
            return 1
        i = 0
        k = 0
        S = []
        for j in range(i + 1, i + nums[i]+1):
            s = j - i + nums[j]
            S.append((j,s))
        S.sort(key=lambda x: x[1], reverse=True)
        k += 1
        if S[0][0] >= len(nums):
            return k
        else:
            k = k + jump(nums[S[0][0]:])
            return k


def ave(i,j,nums):
    return sum(nums[i:j+1])/(j-i+1)
def largestSumOfAverages(nums,k):
    n=len(nums)
    dp=[[0]*(k) for _ in range(n)]
    for i in range(n):
        dp[i][0]=ave(0,i,nums)
        for j in range(1,k):
            for l in range(i):
                dp[i][j]=max(dp[i][j],dp[l][j-1]+ave(l+1,i,nums))
    return dp[-1][-1]

def maxProduct(nums):
    dp=[[0]*2 for _ in range(len(nums))]
    res=nums[0]
    dp[0][0]=nums[0]
    dp[0][1]=nums[0]
    for i in range(1,len(nums)):
        dp[i][0]=max(dp[i-1][0]*nums[i],dp[i-1][1]*nums[i],nums[i])
        dp[i][1]=min(dp[i-1][0]*nums[i],dp[i-1][1]*nums[i],nums[i])
        res=max(res,dp[i][0])
    return res


# def longestMountain(arr):
#     n=len(arr)
#     dp1=[1]*n###以i为结尾的递增序列的最长长度
#     dp2=[1]*n###以i为开头的递减序列的最长长度
#     for i in range(1,n):
#         if arr[i]>arr[i-1]:
#             dp1[i]=dp1[i-1]+1
#     for j in range(n-2,0,-1):
#         if arr[j]>arr[j+1]:
#             dp2[j]=dp2[j+1]+1
#     res=0
#     for k in range(n):
#         if dp1[k]!=1 and dp2[k]!=1:
#             res=max(res,dp1[k]+dp2[k]-1)
#     return res

def minFallingPathSum(matrix) -> int:
    n=len(matrix)
    if n==1:
        return matrix[0][0]
    res=999999
    for i in range(n-2,-1,-1):
        for j in range(n):
            if j==0:
                matrix[i][j]=matrix[i][j]+min(matrix[i+1][j],matrix[i+1][j+1])
                if i==0:
                    res=min(res,matrix[i][j])
            elif j==n-1:
                matrix[i][j]=matrix[i][j]+min(matrix[i+1][j],matrix[i+1][j-1])
                if i==0:
                    res=min(res,matrix[i][j])
            else:
                matrix[i][j]=matrix[i][j]+min(matrix[i+1][j],matrix[i+1][j-1],matrix[i+1][j+1])
                if i==0:
                    res=min(res,matrix[i][j])
    return res

def maxScoreSightseeingPair(values) -> int:
    dp=[0]*len(values)
    res=values[0]+values[1]-1
    for i in range(1,len(values)):
        dp[i]=max(dp[i-1]-values[i-1]+i-1+values[i]-i,values[i]+values[i-1]-1)
        res=max(res,dp[i])
    return res


def videoStitching(clips,time):
    dp = [0] + [float("inf")] * time
    for i in range(1, time + 1):
        for aj, bj in clips:
            if aj < i <= bj:
                dp[i] = min(dp[i], dp[aj] + 1)
    return -1 if dp[time] == float("inf") else dp[time]
'''
并且如果 seq[i+1] - seq[i]( 0 <= i < seq.length - 1) 的值都相同，那么序列 seq 是等差的。
dp[i]表示什么意思
'''


def maxSumTwoNoOverlap(A, L: int,M):
    # L在左边 M在右边
    resL = 0
    resLM = 0
    for i in range(len(A) - L - M + 1):
        resL = max(resL, sum(A[i:i + L]))
        resLM = max(resLM, resL + sum(A[i + L:i + L + M]))

    # L在右边 M在左边
    resM = 0
    resML = 0
    for i in range(len(A) - L - M + 1):
        resM = max(resM, sum(A[i:i + M]))
        resML = max(resML, resM + sum(A[i + M:i + L + M]))

    res = max(resLM, resML)
    return res

def dengcha(nums):
    n=len(nums)
    dp=[{}]*len(nums)
    res=0
    for i in range(1,n):
        for j in range(i):
            k=nums[i]-nums[j]
            dp[i][k]=dp[j].get(k,1)+1
            res=max(res,max(dp[i].values()))
    return res
###dp[i] 表示o-i的最长的等差数列且以nums[i] 为结尾

def feibo(nums):
    n=len(nums)
    dp=[[2]*n for _ in range(n)]
    dic_c={}
    res=0
    for i in range(n):
        dic_c[nums[i]]=i
    for i in range(n-1):
        for j in range(i+1,n):
            if nums[j]-nums[i] in dic_c and dic_c[nums[j]-nums[i]]<i:
                dp[i][j]=dp[dic_c[nums[j]-nums[i]]][i]+1
                res=max(res,dp[i][j])
    return res
###dp[i][j]表设计0-j的最长的feibo数列，且以nums[i],nums[j]为结尾

import  numpy as np
cal=np.array([1,2,3,4])
cal.reshape(1,4)
print(cal)