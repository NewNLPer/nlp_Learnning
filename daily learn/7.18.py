# -*- coding:utf-8 -*-
# @Time      :2022/7/18 14:31
# @Author    :Riemanner
# Write code with comments !!!
def minFlipsMonoIncr(s: str):
    n=len(s)
    dp=[[0]*2 for _ in range(n)]
    if s[0]=='1':
        dp[0][0]=1
    if s[0]=='0':
        dp[0][1]=1
    for i in range(1,n):
        if s[i]=='0':
            dp[i][0]=dp[i-1][0]
            dp[i][1]=min(dp[i-1][0],dp[i-1][1])+1
        else:
            dp[i][0]=dp[i-1][0]+1
            dp[i][1]=min(dp[i-1][0],dp[i-1][1])
    return min(dp[-1])


def minimumTotal(triangle):
    for i in range(len(triangle)-2,-1,-1):
        for j in range(len(triangle[i])):
            triangle[i][j]=min(triangle[i+1][j],triangle[i+1][j+1])+triangle[i][j]
    return triangle[0][0]


# maze=[
#     [1,1,1,1,1,1,1,1,1,1],
#     [1,0,0,1,0,0,0,1,0,1],
#     [1,0,0,1,0,0,0,1,0,1],
#     [1,0,0,0,0,1,1,0,0,1],
#     [1,0,1,1,1,0,1,0,0,1],
#     [1,0,0,0,1,0,0,0,0,1],
#     [1,0,1,0,0,0,1,0,0,1],
#     [1,0,1,0,1,1,1,1,0,1],
#     [1,1,0,0,0,1,0,0,0,1],
#     [1,1,1,1,1,1,1,1,1,1]
# ]
dirs=[
    lambda x,y:(x+1,y),
    lambda x,y:(x-1,y),
    lambda x,y:(x,y-1),
    lambda x,y:(x,y+1)
]
def maze_path(nums):#起点终点坐标
    dirs = [
        lambda x, y: (x + 1, y),
        lambda x, y: (x, y + 1)
    ]
    if nums[0][0]==1:
        return []
    else:
        stack=[]
        stack.append([0,0])
        nums[0][0]=2
        while len(stack)>0:
            curNode=stack[-1]
            if curNode[0]==len(nums)-1 and curNode[1]==len(nums[0])-1:###通过while不断进行迭代，对是否到了终点进行判断
                return stack
            for dir in dirs:
                nextNode=list(dir(curNode[0],curNode[1]))
                if 0<=nextNode[0]<len(nums) and 0<=nextNode[1]<len(nums[0]) and nums[nextNode[0]][nextNode[1]]==0:
                    stack.append(nextNode)
                    nums[nextNode[0]][nextNode[1]]=2#表示已经走过：
                    break
            else:#否则走回头路，再去搜寻
                # maze[nextNode[0]][nextNode[1]]=2
                stack.pop()
        else:
            return []

def strr(s):
    return str(s)

def findRelativeRanks(score):
    dic_c={}
    nums=score[:]
    nums.sort(reverse=True)
    dic_c[nums[0]]= "Gold Medal"
    dic_c[nums[1]]="Silver Medal"
    dic_c[nums[2]]="Bronze Medal"
    for i in range(3,len(nums)):
        dic_c[nums[i]]=str(i+1)
    for i in range(len(score)):
        score[i]=dic_c[score[i]]
    return score


def waysToChange(n: int) -> int:
    dp=[0]*(n+1)
    dp[0]=1
    for i in range(1,n+1):
        if i-1>=0:
            dp[i]+=(dp[i-1]%1000000007)
        if i-5>=0:
            dp[i]+=(dp[i-5]%1000000007)
        if i-10>=0:
            dp[i]+=(dp[i-10]%1000000007)
        if i-25>=0:
            dp[i]+=(dp[i-25]%1000000007)
    print(dp)
    return dp[-1]

def pileBox1(box:list):
    n=len(box)
    box.sort(key=lambda x:[x[0],x[1]],reverse=True)
    dp=[0]*n
    dp[0]=box[0][2]
    res=0
    for i in range(1,n):
        for j in range(i):
            if box[i][0]<box[j][0] and box[i][1]<box[j][1] and box[i][2]<box[j][2]:
                dp[i]=max(dp[i],dp[j]+box[i][2])
                res=max(res,dp[i])
    return res

from functools import lru_cache
####回溯算法
# def pileBox(box:list):
#     def bt(cc,box):
#         res=0
#         for j in range(len(box)):
#             if cc[0]<box[j][0] and cc[1]<box[j][1] and cc[2]<box[j][2]:
#                 res=max(res,box[j][2]+bt(box[j],box[:j]+box[j+1:]))
#         return res
#     res1=0
#     for i in range(len(box)):
#         res1=max(res1,box[i][2]+bt(box[i],box[:i]+box[i+1:]))
#     return res1
###记忆化算法
def pileBox(box:list):
    dic_c={}
    def bt(cc,box,dic):
        if tuple(cc) in dic:
            return dic[tuple(cc)]
        else:
            res=cc[0]
            for j in range(len(box)):
                if cc[0]<box[j][0] and cc[1]<box[j][1] and cc[2]<box[j][2]:
                    res=max(res,res+bt(box[j],box[:j]+box[j+1:],dic_c))
            dic_c[tuple(cc)]=res
            return res
    res1=0
    for i in range(len(box)):
        res1=max(res1,bt(box[i],box[:i]+box[i+1:],dic_c))
    print(dic_c)
    return res1


def minHeightShelves(books, shelf_width):
    n = len(books)
    dp = [1000000] * (len(books)+1)
    dp[0] = 0
    for i in range(1, n + 1):
        tmp_width, j, h = 0, i, 0
        while j > 0:
            tmp_width += books[j - 1][0]
            if tmp_width > shelf_width:
                break
            h = max(h, books[j - 1][1])
            dp[i] = min(dp[i], dp[j - 1] + h)
            j -= 1
    print(dp)
    return dp[-1]

def shiftGrid(grid,k):
    m=len(grid)
    n=len(grid[0])
    path=[]
    for item in grid:
        path+=item
    k=k%(m*n)
    path=path[len(path)-k:]+path[:len(path)-k]
    res=[]
    c=0
    while c!=len(path):
        path1=path[c:c+n]
        res.append(path1)
        c=c+n
    return res


def lengthOfLongestSubstring(s):
    if len(s)==0:
        return 0
    else:
        start_zhen=0
        end_zhen=1
        s1=set()
        s1.add(s[start_zhen])
        res=0
        while end_zhen<len(s):
            if s[end_zhen] not in s1:
                s1.add(s[end_zhen])
                end_zhen+=1
                res=max(res,len(s1))
            else:
                s1.add(s[end_zhen])
                start_zhen+=1
                end_zhen+=1
                res = max(res, len(s1))
        return res
print(lengthOfLongestSubstring(s = "pwwkew"))