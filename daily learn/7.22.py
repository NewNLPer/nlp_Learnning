# -*- coding:utf-8 -*-
# @Time      :2022/7/22 14:29
# @Author    :Riemanner
# Write code with comments !!!
def intersectionSizeTwo(intervals:list):
    intervals.sort(key=lambda x:[x[0],-x[1]])
    cur=intervals[-1][0]
    next=intervals[-1][0]+1
    geshu=2
    for i in range(len(intervals)-1,-1,-1):
        if intervals[i][1]<cur:
            geshu+=2
            cur=intervals[i][0]
            next=intervals[i][0]+1
        elif next<=intervals[i][1]:
            continue
        elif cur<=intervals[i][1]<next:
            geshu+=1
            next=cur
            cur=intervals[i][0]
    return geshu

def countVowelStrings(n):
    ###dp[i][j]以其为结尾的字母的个数
    dp=[[0]*5 for _ in range(n)]
    dp[0][0]=1####以a为结尾
    dp[0][1]=1####以e为结尾
    dp[0][2]=1####以i为结尾
    dp[0][3]=1####以o为结尾
    dp[0][4]=1####以u为结尾
    for i in range(1,n):
        dp[i][0]=dp[i-1][0]
        dp[i][1]=dp[i-1][0]+dp[i-1][1]
        dp[i][2]=dp[i-1][0]+dp[i-1][1]+dp[i-1][2]
        dp[i][3]=dp[i-1][0]+dp[i-1][1]+dp[i-1][2]+dp[i-1][3]
        dp[i][4]=dp[i-1][0]+dp[i-1][1]+dp[i-1][2]+dp[i-1][3]+dp[i-1][4]
    return sum(dp[-1])


def maxSubarraySumCircular(nums):
    def dpp1(nums:list):####求连续的最大值
        if len(nums)==1:
            return nums[0]
        else:
            n=len(nums)
            dp=[0]*len(nums)
            dp[0]=nums[0]
            res=nums[0]
            for i in range(1,n):
                dp[i]=max(nums[i],dp[i-1]+nums[i])
                res=max(res,dp[i])
            return res
    def dpp2(nums:list):###求连续的最小值
        if len(nums)==1:
            return nums[0]
        else:
            n=len(nums)
            dp=[0]*n
            dp[0]=nums[0]
            res=nums[0]
            for i in range(1,n):
                dp[i]=min(dp[i-1]+nums[i],nums[i])
                res=min(res,dp[i])
            return res
    if len(nums)==1:
        return nums[0]
    else:
        res=sum(nums)-dpp2(nums[1:len(nums)-1])
        return max(res,dpp1(nums))


def findBall(grid):
    n = len(grid[0])
    ans = [-1] * n
    for j in range(n):
        col = j  # 球的初始列
        for row in grid:
            dir = row[col]
            col += dir  # 移动球
            if col < 0 or col == n or row[col] != dir:  # 到达侧边或 V 形
                break
        else:  # 成功到达底部
            ans[j] = col
    return ans

import collections
def sequenceReconstruction1111111111(nums,sequences):
    n = len(nums)
    adj = [set() for _ in range(n + 1)]
    degree = [0 for _ in range(n + 1)]
    for sequence in sequences:
        for i in range(1, len(sequence)):
            adj[sequence[i - 1]].add(sequence[i])
            degree[sequence[i]] += 1
    q = collections.deque()
    for i in range(1, n + 1):
        if degree[i] == 0:
            q.append(i)
    while q:
        if len(q) > 1:
            return False
        i = q.popleft()
        for j in adj[i]:
            degree[j] -= 1
            if degree[j] == 0:
                q.append(j)
    return True



def sequenceReconstruction(nums,sequences):
    l = len(nums)
    d = {}#哈希表
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] not in d:
                d[seq[i]] = set()
            d[seq[i]].add(seq[i + 1])
    for i in range(l - 1):#检查每个元素是否出现在前一个元素的哈希表值中,nums是唯一最短超序列的充要条件就是nums中每一个元素的下一个元素都在哈希表中出现过.
        if nums[i] not in d or nums[i + 1] not in d[nums[i]]:
            return False
    return True

def distanceBetweenBusStops(distance,start,destination):
    if start<destination:
        return min(sum(distance[start:destination]),sum(distance[destination:])+sum(distance[:start]))
    else:
        start1,destination1=destination,start
        return min(sum(distance[start1:destination1]),sum(distance[destination1:])+sum(distance[:start1]))


def equalPairs(grid):
    ####生成这个矩阵的所有列，然后再进行比较
    grid1=[]
    for i in range(len(grid[0])):
        path=[]
        for j in range(len(grid)):
            path.append(grid[j][i])
        grid1.append(path)
    res=0
    for item1 in grid:
        for item2 in grid1:
            if item1[0]!=item2[0]:
                continue
            elif item1==item2:
                res+=1
    return res

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class CBTInserter:
    def __init__(self, root: TreeNode):
        self.root=root
        self.path=[self.root]
        self.path1=[self.root]
        while self.path1:
            self.path2=[]
            for node in self.path1:
                if node.left:
                    self.path2.append(node.left)
                    self.path.append(node.left)
                if node.right:
                    self.path2.append(node.right)
                    self.path.append(node.right)
            self.path1=self.path2
    def insert(self, val: int) -> int:
        for node in self.path:
            if not node.left and not node.right:
                node.left=TreeNode(val)
                self.path.append(node.left)
                return node.val
            elif not node.right:
                node.right=TreeNode(val)
                self.path.append(node.right)
                return node.val
    def get_root(self) -> TreeNode:
        return self.root

import torch
x=torch.randn(4,3)
print(x)
print(x.shape)
x=x.view(12)
print(x.shape)
print(x)


