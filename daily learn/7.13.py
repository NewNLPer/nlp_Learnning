# -*- coding:utf-8 -*-
# @Time      :2022/7/13 8:57
# @Author    :Riemanner
# Write code with comments !!!
def asteroidCollision(asteroids):
    res=[]
    for item in asteroids:
        if not res:
            res.append(item)
        elif res[-1]>0 and item<0:
            while res and res[-1]>0 and item<0:
                cc=1
                if abs(res[-1])>abs(item):
                    cc=0
                    break
                elif abs(res[-1])==abs(item):
                    cc=0
                    res.append(item)
                    res.pop()
                    res.pop()
                    break
                else:
                    res.pop()
            if cc==1:
                res.append(item)
        else:
            res.append(item)
    return res

def f(pref: str, suff: str):
    dic_c={'a':[('apple',0)]}
    n1=len(pref)
    n2=len(suff)
    res=-1
    if dic_c.get(pref[0],0)==0:
        return res
    else:
        for item in dic_c[pref[0]]:
            if item[0][:n1]==pref and item[0][len(item[0])-n2:]==suff:
                res=item[1]
        return res


def minCostClimbingStairs(cost):
    cost.append(0)
    n=len(cost)
    dp=[0]*n
    dp[0]=cost[0]
    dp[1]=cost[1]
    for i in range(2,n):
        dp[i]=min(dp[i-2],dp[i-1])+cost[i]
    return dp[-1]


def updateMatrix(mat):
    mat1=mat[:]
    n1=len(mat)
    n2=len(mat[0])
    dp=[[0]*n2 for _ in range(n1)]
    def bfs(i,j):
        ans=9999999999999999
        s=[(1,0),(0,1),(-1,0),(0,-1)]
        for x,y in s:
            if 0<=i+x<n1 and 0<=j+y<n2:
                if mat[i+x][j+y]==1:
                    ans=min(ans,1+bfs(i+x,j+y))
                elif mat[i+x][j+y]==0:
                    ans=min(ans,1)
        return ans
    for i in range(n1):
        for j in range(n2):
            if mat[i][j]==1:
                dp[i][j]=bfs(i,j)
    return dp

def numberOfPairs(nums):
    dic_c={}
    for num in nums:
        dic_c[num]=dic_c.get(num,0)+1
    res1=0###表示对数
    res2=0###表示剩的数
    for key in dic_c:
        if dic_c[key]>=2:
            res1+=(dic_c[key]//2)
            res2+=(dic_c[key]%2)
        else:
            res2+=1
    return [res1,res2]



def maximumSum(nums):
    dic_c={}
    res=-1
    for item in nums:
        sum1=0
        for it in str(item):
            sum1+=int(it)
        dic_c[sum1]=dic_c.get(sum1,[])+[item]
        if len(dic_c[sum1])>2:
            mins=min(dic_c[sum1])
            dic_c[sum1].remove(mins)
    for key in dic_c:
        if len(dic_c[key])==2:
            res=max(res,sum(dic_c[key]))
    return res

import math
def gcd_many(s):
    g = 0
    for i in range(len(s)):
        if i == 0:
            g = s[i]
        else:
            g=math.gcd(g,s[i])
    return g
def minOperations(nums,numsDivide):
    yue_shu=gcd_many(numsDivide)
    dic_c={}
    res=0
    for item in nums:
        dic_c[item]=dic_c.get(item,0)+1
    nums=list(set(nums))
    nums.sort()
    for each in nums:
        if yue_shu%each==0:
            return res
        else:
            res+=dic_c[each]
    return -1




def smallestTrimmedNumbers(nums,queries):
    m, n, sz = len(nums), len(queries), len(nums[0])
    arr = list(range(m))
    ret = [0] * n
    for i in range(n):
        k, t = queries[i]
        arr.sort(key = lambda x: (nums[x][-t:], x))
        print(arr)
        ret[i] = arr[k - 1]
    return ret

def arrayNesting(nums):
    res=1
    for i in range(len(nums)):
        if nums[i]!='*':
            num=nums[i]
            nums[i]='*'
            a=nums[num]
            res1=1
            while a!='*':
                res1+=1
                b=a
                a=nums[a]
                nums[b]='*'
            res=max(res,res1)
            if res>len(nums)//2:
                return res
    return res

def rob(nums):
    if len(nums)==1:
        return nums[0]
    if len(nums)==2:
        return max(nums)
    else:
        def dpi(nums):
            n=len(nums)
            dp=[0]*n
            dp[0]=nums[0]
            dp[1]=max(nums[0],nums[1])
            for i in range(2,n):
                dp[i]=max(dp[i-1],dp[i-2]+nums[i])
            return dp[-1]
    return max(dpi(nums[1:]),dpi(nums[:len(nums)-1]))

def minCost(costs):
    n1=len(costs)
    n2=len(costs[0])
    dp=[[0]*n2 for _ in range(n1)]###dp[i][0,1,2]表示第i个房子刷成什么颜色的最低费用
    dp[0][0]=costs[0][0]
    dp[0][1]=costs[0][1]
    dp[0][2]=costs[0][2]
    for i in range(1,n1):
        dp[i][0]=min(dp[i-1][1],dp[i-1][2])+costs[i][0]
        dp[i][1] =min(dp[i-1][0],dp[i-1][2])+costs[i][1]
        dp[i][2]=min(dp[i-1][1],dp[i-1][0])+costs[i][2]
    return min(dp[-1])
print(minCost( costs = [[7,6,2]]))

