# -*- coding:utf-8 -*-
# @Time      :2022/7/6 15:22
# @Author    :Riemanner
# Write code with comments !!!
def minScoreTriangulation(values):
    values.sort()
    res=0
    c=1
    for j in range(3):
      c*=values[j]
    res+=c
    values.pop(2)
    while len(values)>=3:
        c=1
        for i in range(3):
           c*=values[i]
        res+=c
        values.pop(2)
    return res

def largestSumOfAverages(nums,k):
    n=len(nums)
    dp=[[0]*(k) for _ in range(n)]
    for i in range(n):###字符
        dp[i][0]=sum(nums[0:i+1])
        for j in range(1,k):###分割次数
            for l in range(i):
                dp[i][j]=max(dp[i][j],dp[l][j-1]+sum(nums[l+1:i+1]))
    return dp[-1][-1]

def maxSumAfterPartitioning(arr,k):
    '''
    dp[i]表示i-1索引处作为分割数组的点，能得到的arr[:i]最大元素和。
    在i-1处分割，上一个分割的数组长度可能为1~k，遍历所有可能的结果得到最大元素和。
    dp[i] = max(dp[i] ,dp[i-j] + max(arr[i-j:i])*j)  j=1~k
    '''
    n = len(arr)
    dp = [0]*(n+1)
    dp[1]=arr[0]
    for i in range(2, n+1):
        for j in range(i):
            if i-j<=k:
                dp[i] = max(dp[i], dp[j] + max(arr[j:i])*(i-j))
    return dp[n]


def star_pan(s1,s2):
    if abs(len(s1)-len(s2))>=2 or abs(len(s1)-len(s2))==0:
        return False
    else:
        start_zhen1=0
        start_zhen2=0
        while start_zhen1<len(s1) and start_zhen2<len(s2):
            if s1[start_zhen1]==s2[start_zhen2]:
                start_zhen1+=1
                start_zhen2+=1
            else:
                start_zhen2+=1
        return start_zhen1==len(s1)

def longestStrChain(words:list):
    words.sort(key=lambda x:len(x))
    n=len(words)
    dp=[1]*n
    res=1
    for i in range(1,n):
        for j in range(i):
            if star_pan(words[j],words[i]):
                dp[i]=max(dp[i],dp[j]+1)
                res=max(res,dp[i])
    return res


###最长feibonaqie


def lenLongestFibSubseq(arr) -> int:
    num2idx = {x : i for i, x in enumerate(arr)}###元素——>索引的映射
    n = len(arr)
    # arr[i] -> arr[j], only prev_idx < i < j is valid
    dp = [[2] * n for _ in range(n)]
    ans = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            prev = arr[j] - arr[i]
            if prev in num2idx and num2idx[prev] < i:
                prev_idx = num2idx[prev]
                dp[i][j] = dp[prev_idx][i] + 1
                ans = max(ans, dp[i][j])
    return ans

def function11(nums):
    dp=[1]*len(nums)
    res=0
    for i in range(1,len(nums)):
        for j in range(i):
            if nums[i]>nums[j]:
                dp[i]=max(dp[j]+1,dp[i])
        if dp[i]>=3:
            res+=(dp[i]-3+1)
    return res


def numOfSubarrays(arr):
    n=len(arr)
    dp=[[0]*2 for _ in range(n)]
    res=0
    for i in range(n):
        if arr[i]%2==0:
            dp[i][0]=dp[i-1][0]
            dp[i][1]=dp[i-1][1]+1
        else:
            dp[i][0]=dp[i-1][1]+1
            dp[i][1]=dp[i-1][0]
        res+=dp[i][0]
    return res


def longestSubsequence(arr,difference):
    n=len(arr)
    dp=[1]*n
    res=0
    if arr[1]-arr[0]==difference:
        dp[1]=2
    for i in range(2,n):
        s=set(arr[:i])
        if arr[i]-difference not in s:
            continue
        else:
            for j in range(i):
                if arr[i]-arr[j]==difference:
                    dp[i]=max(dp[i],dp[j]+1)
                    res=max(res,dp[i])
    return max(res,1,dp[1])
s='101110010101011'
c=s.split('0')
def longestSubarray(nums):
    if sum(nums)==len(nums):
        return sum(nums)-1
    for i in range(len(nums)):
        nums[i]=str(nums[i])
    c=''.join(nums).split('0')
    res=0
    for i in range(len(c)):
        if c[i]==' ':
            continue
        elif i==0 and c[i+1]!=' ':
            res=max(res,len(c[i])+len(c[i+1]))
        elif i==0 and c[i+1]==' ':
            res=max(res,len(c[i]))
        elif i==len(c)-1 and c[i-1]!=' ':
            res=max(res,len(c[i])+len(c[i-1]))
        elif i==len(c)-1 and c[i-1]==' ':
            res=max(res,len(c[i]))
        elif c[i]!=' ' and c[i-1]!=' ' and c[i+1]!=' ':
            res=max(res,len(c[i])+len(c[i-1]),len(c[i])+len(c[i+1]))
        elif c[i]!=' ' and c[i-1]==' ' and c[i+1]!=' ':
            res = max(res,len(c[i])+len(c[i+1]))
        elif c[i]!=' ' and c[i-1]!=' ' and c[i+1]==' ':
            res = max(res,len(c[i])+len(c[i-1]))
    return res