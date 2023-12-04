# -*- coding:utf-8 -*-
# @Time      :2022/6/23 14:17
# @Author    :Riemanner
# Write code with comments !!!
###开始进行动态规划的学习
# def minCostClimbingStairs(cost):
#     dp=[0]*(len(cost)+1)
#     dp[0]=0
#     dp[1]=0
#     for i in range(2,len(cost)+1):
#         dp[i]=min(dp[i-2]+cost[i-2],dp[i-1]+cost[i-1])
#     print(dp)
#     return dp[-1]
# print(minCostClimbingStairs(cost = [10,15,20]))
def uniquePathsWithObstacles(obstacleGrid):
    m=len(obstacleGrid)
    n=len(obstacleGrid[0])
    dp=[]
    for i in range(m):
        dp.append([0]*n)
    dp[0][0]=1
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j]==1:
                dp[i][j]=0
            else:
                if i==0 and j==0:
                    continue
                elif i-1<0:
                    dp[i][j]=dp[i][j-1]
                elif j-1<0:
                    dp[i][j]=dp[i-1][j]
                else:
                    dp[i][j]=dp[i-1][j]+dp[i][j-1]
    return dp[-1][-1]


def integerBreak(n):
    dp=[0]*(n+1)
    dp[2]=1
    for i in range(3,n+1):
        for j in range(1,i):
            dp[i]=max(j*dp[i-j],dp[i],j*(i-j))
    return dp[n]

###0-1背包问题
def function(weight,value,n):
    dp=[]
    for i in range(len(weight)):
        dp.append([0]*(n+1))
    for i in range(len(weight)):
        for j in range(n+1):
            if i==0 and weight[i]<=j:
                dp[i][j]=value[i]
            elif j==0:
                dp[i][j]=0
            else:
                if j<weight[i]:
                    dp[i][j]=dp[i-1][j]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i])



def lastStoneWeightII(stones):
    jk=sum(stones)
    junzhi=int(jk/2)
    dp=[]
    for i in range(len(stones)):
        dp.append([0]*(junzhi+1))
    for j in range(len(stones)):
        for k in range(junzhi+1):
            if j==0 and stones[j]<=k:
                dp[j][k]=stones[j]
            elif k==0:
                dp[j][k]=0
            else:
                if k<stones[j]:
                    dp[j][k]=dp[j-1][k]
                else:
                    dp[j][k]=max(dp[j-1][k],dp[j-1][k-stones[j]]+stones[j])
    return max(jk-dp[-1][-1],dp[-1][-1])-min(jk-dp[-1][-1],dp[-1][-1])


# def findMaxForm(strs,m,n):
#     dp = [[0] * (n + 1) for _ in range(m + 1)]	# 默认初始化0
#     # 遍历物品
#     for str in strs:
#         ones = str.count('1')
#         zeros = str.count('0')
#         # 遍历背包容量且从后向前遍历！
#         for i in range(m, zeros - 1, -1):
#             for j in range(n, ones - 1, -1):
#                 dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
#     return dp[m][n]
# print(findMaxForm(['10'],3,3))
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

a=ListNode(1)
b=ListNode(2)
c=ListNode(4)
a.next=b
b.next=c

d=ListNode(1)
e=ListNode(3)
f=ListNode(4)
d.next=e
e.next=f

def mergeTwoLists(l1,l2):
    if not l1:
        return l2
    elif not l2:
        return l1
    else:
        tail1=l1
        tail2=l2
        head=ListNode(999)
        tail3=head
        while  tail1 and  tail2:
            if tail1.val<=tail2.val:
                tail3.next=tail1
                tail3=tail3.next
                tail1=tail1.next
            else:
                tail3.next=tail2
                tail3=tail3.next
                tail2=tail2.next
        if not tail1:
            tail3.next=tail2
        else:
            tail3.next=tail1
        return head.next
c=mergeTwoLists(a,d)


from functools import lru_cache
def divide(dividend,divisor):
    if divisor==1 and dividend>=0:
        return min(dividend,pow(2,31)-1)
    elif divisor==1 and dividend<0:
        return max(dividend,-1*pow(2,31))
    elif divisor==-1 and dividend>=0:
        return max(-dividend,-1*pow(2,31))
    elif divisor==-1 and dividend<0:
        return min((-1)*dividend,pow(2,31)-1)
    else:
        @lru_cache()
        def function(n1,n2):
            if n1==0:
                return 0
            elif n1>0 and n2>n1:
                return 0
            else:
                s=1
                while n2+n2<=n1:
                    n2=n2+n2
                    s=s+s
                return s+function(n1-n2,n2)
        return (function(abs(dividend),abs(divisor)))*(dividend//(abs(dividend)))*(divisor//(abs(divisor)))





###0-1背包
def beibao(weight,values,n):
    dp = [[0] * (n + 1) for _ in range(len(weight))]
    for i in range(len(values)):
        for j in range(n+1):
            if i==0 and weight[i]<=j:
                dp[i][j]=values[i]
            elif j==0:
                dp[i][j]=0
            elif j-weight[i]<0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + values[i])
    return dp[-1][-1]


import math
def he(n):
    dp=[n+1]*(n+1)
    dp[0]=0
    for i in range(0,n+1):
        for j in range(1,int(math.sqrt(i))+1):
            dp[i]=min(dp[i],dp[i-j*j]+1)
    print(dp)
    return dp[-1]






###完全背包
def beibao2(weight,values,n):
    dp = [[0] * (n + 1) for _ in range(len(weight))]
    for j in range(n+1):
        for i in range(len(weight)):
            if j==0 and weight[i]<=j:
                dp[i][j]=values[i]
            elif j==0:
                dp[i][j]=0
            elif j-weight[i]<0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + values[i])
    print(dp)
    return dp[-1][-1]


def qiege(s,w):
    dp=[False]*(len(s)+1)
    dp[0]=True
    for i in range(1,len(s)+1):
        for j in range(i):
            s1=s[j:i]
            if s1 in set(w) and dp[j]:
                dp[i]=True
    print(dp)
    return dp[-1]


def lingqian(coins,amount):
    dp=[999999999]*(amount+1)
    dp[0]=0
    for i in range(1,amount+1):
        for j in range(len(coins)):
            if i-coins[j]>=0 and dp[i-coins[j]]!=999999999:
                dp[i]=min(dp[i],dp[i-coins[j]]+1)
    print(dp)
    return dp[-1]


def change1(nums,target):
    dp=[0]*(target+1)
    dp[0]=1
    for i in range(len(nums)):
        for j in range(target+1):
            if j>=nums[i]:
                dp[j]+=dp[j-nums[i]]
    print(dp)
    return dp[-1]


def change3(nums,target):
    dp = [0] * (target + 1)
    dp[0] = 1
    for i in range(target+1):
        for j in range(len(nums)):
            if i>=nums[j]:
                dp[i]+=dp[i-nums[j]]
    return dp[-1]


def function6(s):
    if len(str(s))==1:
        return s
    elif len(str(s))==2:
        return int(str(s)[0])+int(str(s)[1])
    else:
        return int(str(s)[0])+int(str(s)[1])+int(str(s)[2])

def movingCount(m,n,k):
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if function6(i) + function6(j) <= k:
                dp[i][j] = 1

    def dfs(x, y):
        if dp[x][y] == 0 or dp[x][y] == '#':
            return
        else:
            dp[x][y] = '#'
        for c in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            if 0 <= x + c[0] < m and 0 <= y + c[1] < n:
                dfs(x + c[0], y + c[1])

    dfs(0, 0)
    res = 0
    for i in range(m):
        for j in range(n):
            if dp[i][j] == '#':
                res += 1
    return res

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
a=TreeNode(1)
b=TreeNode(2)
c=TreeNode(2)
d=TreeNode(2)
e=TreeNode(2)

a.left=b
a.right=c
b.right=d
c.right=e

def zhongxu(root):
    if not root:
        return ['*']
    elif root and not root.left and not root.right:
        return [root.val]
    else:
        return zhongxu(root.left)+[root.val]+zhongxu(root.right)

def myPow(x,n):
    if n==1:
        return x
    elif n==-1:
        return 1/x
    elif x==0:
        return 0
    elif n==0:
        return 1
    else:
        n1=abs(x)
        n2=1
        while 2*n2<=abs(n):
            n2=n2+n2
            n1=n1*n1
        c=n1*myPow(x,abs(n)-n2)
        if x>0 and n>0:
            return c
        elif x>0 and n<0:
            return 1/c
        elif x<0 and n<0:
            if n%2==0:
                return 1/c
            else:
                return -1/c
        elif x<0 and n>0:
            if n%2==0:
                return c
            else:
                return -1*c


def function33(nums):
    dp=[0]*(len(nums))
    dp[0]=nums[0]
    for i in range(1,len(nums)):
        dp[i]=max(dp[i-1],dp[i-1]+nums[i],nums[i])
    print(dp)
    return dp[-1]


def function22(pushed,popped):
    try:
        zhan_p=[]
        start_zhen=0
        end_zhen=0
        while start_zhen<len(pushed) or end_zhen<len(popped):
            if len(zhan_p)==0:
                zhan_p.append(pushed[start_zhen])
                start_zhen+=1
            elif zhan_p[-1]!=popped[end_zhen]:
                zhan_p.append(pushed[start_zhen])
                start_zhen+=1
            elif zhan_p[-1]==popped[end_zhen]:
                zhan_p.pop()
                end_zhen+=1
    except(ValueError):
        return False
    else:
        return True



def erchashu(root):
    res=[]
    def bt(root,path):
        if not root:
            return
        if root and not root.left and not root.right:
            res.append(path+[root.val])
        bt(root.left,path+[root.val])
        bt(root.right,path+[root.val])
    bt(root,[])
    return res

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
a=TreeNode(1)
b=TreeNode(2)
c=TreeNode(3)
d=TreeNode(4)
e=TreeNode(5)



def minNumber(nums):
    n=len(nums)
    while n>0:
        for i in range(len(nums)-1):
            s1=str(nums[i])+str(nums[i+1])
            s2=str(nums[i+1])+str(nums[i])
            if int(s1)>int(s2):
                nums[i],nums[i+1]=nums[i+1],nums[i]
        n-=1
    for j in range(len(nums)):
        nums[j]=str(nums[j])
    return ''.join(nums)


###递归函数:

###初值 F(0)=0,F(1)=1
###递推公式  F(n)=F(n-1)+F(n-2)
def functionss(s):
    s=str(s)
    dp=[0]*len(s)
    dp[0]=1
    if int(s[:2])<=26:
        dp[1]=2
    else:
        dp[1]=1
    for i in range(2,len(s)):
        if int(s[i-1:i+1])>26:
            dp[i]=dp[i-1]
        elif str(int(s[i-1:i+1]))!=s[i-1:i+1]:
            dp[i]=dp[i-3]
        else:
            dp[i]=dp[i-1]+dp[i-2]
    return dp[-1]
print(functionss(0))