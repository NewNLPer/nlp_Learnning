# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/1/15 10:57
coding with comment！！！
"""
'''
快排O(nlogn)
'''




def quick_sort(nums,left,right):
    def take_nums(nums,left,right):
        tmp=nums[left]
        while left<right:
            while left<right and nums[right]>=tmp:
                right-=1
            nums[left],nums[right]=nums[right],nums[left]
            while left<right and nums[left]<=tmp:
                left+=1
            nums[left],nums[right]=nums[right],nums[left]
        nums[left]=tmp
        return left
    if left<right:
        mid=take_nums(nums,left,right)
        quick_sort(nums,left,mid-1)
        quick_sort(nums,mid+1,right)
        return nums

'''
堆排序，的topk问题###O(k+nlog(k));
前k个最大的数，那就是小根堆+遍历数组的时候选择>
前k个最小的数，那就是小根堆+遍历数组的时候选择<
'''


def topk(nums,k):
    def sift(nums,left,high):
        tmp=nums[left]
        i=left
        j=2*i+1
        while j<=high:
            if j+1<=high and nums[j+1]<nums[j]:
                j=j+1
            if nums[j]<tmp:
                nums[i]=nums[j]
                i=j
                j=2*1+1
            else:
                break
        nums[i]=tmp
    head=nums[:k]
    for i in range((k-2)//2,-1,-1):
        sift(head,i,k-1)
    for i in range(k,len(nums)):
        if nums[i]>head[0]:
            head[0]=nums[i]
            sift(head,0,k-1)
    for i in range(k-1,-1,-1):
        head[i],head[0]=head[0],head[i]
        sift(head,0,i-1)
    return head




def tok_k(nums,k):
    def sift(nums,low,high):###堆的向下调整，其满足只有根节点不满足堆，两边都满足
        tmp=nums[low]
        i=low
        j=2*i+1
        while j<=high:
            if j+1<=high and nums[j+1]<nums[j]:
                j=j+1
            if nums[j]<tmp:
                nums[i]=nums[j]
                i=j
                j=2*i+1
            else:
                break
        nums[i]=tmp

    head=nums[:k]
    for i in range((k-2)//2,-1,-1):#对head进行建堆(应该思考是建立大根堆还是小根堆)
        sift(head,i,k-1)
    for i in range(k,len(nums)):#遍历数组，最小还是最大
        if nums[i]>head[0]:
            head[0]=nums[i]
            sift(head,0,k-1)
    for i in range(k-1,-1,-1):#出数，按照顺序出数
        head[0],head[i]=head[i],head[0]
        sift(head,0,i-1)
    return head[-1]

nums1=[1332802,1177178,1514891,871248,753214,123866,1615405,328656,1540395,968891,1884022,252932,1034406,1455178,821713,486232,860175,1896237,852300,566715,1285209,1845742,883142,259266,520911,1844960,218188,1528217,332380,261485,1111670,16920,1249664,1199799,1959818,1546744,1904944,51047,1176397,190970,48715,349690,673887,1648782,1010556,1165786,937247,986578,798663]
print(tok_k(nums1,24))

'''
子集问题就是涉及到重复与不重复的问题
就一句话 先sort一下
if start==i or nums[i]!=nums[i-1]:
'''
def subset(nums):##无重复数组
    res=[]
    def bt(path,start):
        res.append(path[:])
        for i in range(start,len(nums)):
            path.append(nums[i])
            bt(path,i+1)
            path.pop()
    bt([],0)
    return res

def re_subset(nums):##重复数组
    nums.sort(reverse=False)
    res=[]
    def bt(path,start):
        res.append(path[:])
        for i in range(start,len(nums)):
            if start==i or nums[i]!=nums[i-1]:
                path.append(nums[i])
                bt(path,i+1)
                path.pop()
    bt([],0)
    return res

'''
排列组合问题
也分为不重复与重复问题
加一句:
sort一下， if i==0 or nums[i-1]!=nums[i]:
'''
def pailie(nums):### 无重复数组
    res=[]
    n=len(nums)
    def bt(path,nums):
        if len(path)==n:
            res.append(path[:])
            return
        for i in range(len(nums)):
            path.append(nums[i])
            bt(path,nums[:i]+nums[i+1:])
            path.pop()
    bt([],nums)
    return res

def re_pailie(nums):### 有重复数组
    res=[]
    n=len(nums)
    nums.sort()
    def bt(path,nums):
        if len(path)==n:
            res.append(path[:])
            return
        for i in range(len(nums)):
            if i==0 or nums[i-1]!=nums[i]:
                path.append(nums[i])
                bt(path,nums[:i]+nums[i+1:])
                path.pop()
    bt([],nums)
    return res

'''
经典的切割问题，处理点就在if p==p[::-1]
'''
def qiege(s):### 回文子串的切割
    res=[]
    def bt(path,start):
        if start>=len(s):
            res.append(path[:])
            return
        for i in range(start,len(s)):
            p=s[start:i+1]
            if p==p[::-1]:
                path.append(p)
                bt(path,i+1)
                path.pop()
            else:
                continue
    bt([],0)
    return res


'''
二分查找，其实就是两种形式的变化
但是要注意前提为，数组为排序数组
'''

def two_search1(nums,target):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<=end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]==target:
            return mid
        elif nums[mid]>target:
            end_zhen=mid-1
        else:
            start_zhen=mid+1
    return -1,start_zhen

def two_search2(nums,target):
    start_zhen=0
    end_zhen=len(nums)
    while start_zhen<end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]==target:
            return mid
        elif nums[mid]>target:
            end_zhen=mid
        else:
            start_zhen=mid+1
    return -1,start_zhen

'''
二分
'''

####关于二分查找
###左闭右闭(无重复元素)
def function1(nums,target):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<=end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]>target:
            end_zhen=mid-1
        elif nums[mid]<target:
            start_zhen=mid+1
        else:
            return mid
    return -1,start_zhen



def function2(nums,target):
    start_zhen=0
    end_zhen=len(nums)
    while start_zhen<end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]<target:
            start_zhen=mid+1
        elif nums[mid]>target:
            end_zhen=mid
        else:
            return mid
    return -1


def get_r(nums,target):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<=end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]>target:
            end_zhen=mid-1
        elif nums[mid]<target:
            start_zhen=mid+1
        else:
            start_zhen=mid+1
    if end_zhen<0 or nums[end_zhen]!=target:
        return -1
    return end_zhen


def ziji1(nums):
    res=[]
    n=len(nums)
    def bt(path,nums):
        if len(path)==n:
            res.append(path[:])
        for i in range(len(nums)):
            if i==0 or nums[i]!=nums[i-1]:
                path.append(nums[i])
                bt(path,nums[:i]+nums[i+1:])
                path.pop()
    bt([],nums)
    return res


def huiwenqiege(s):
    res=[]
    def bt(start,path):
        if start>=len(s):
            res.append(path[:])
            return
        for i in range(start,len(s)):
            p=s[start:i+1]
            if p==p[::-1]:
                path.append(p)
            else:
                continue
            bt(i+1,path)
            path.pop()
    bt(0,[])
    return res



#1.快速排序
def quick_sort1(nums,left,right):
    def take_nums(nums,left,right):
        tmp=nums[left]
        while left<right:
            while left<right and nums[right]>=tmp:
                right-=1
            nums[left],nums[right]=nums[right],nums[left]
            while left<right and nums[left]<=tmp:
                left+=1
            nums[left], nums[right] = nums[right], nums[left]
            nums[left]=tmp
            return left
    if left<right:
        mid=take_nums(nums,left,right)
        quick_sort(nums,left,mid-1)
        quick_sort(nums,mid+1,right)
    return nums

#2.top问题
def top_k1(nums,k):
    def sift(nums,low,high):###两边都是堆，只有根节点不满足堆的定义
        tmp=nums[low]
        i=low
        j=2*i+1
        while j<=high:
            if j+1<=high and nums[j+1]<nums[j]:
                j=j+1
            if nums[j]<tmp:
                nums[i]=nums[j]
                i=j
                j=2*i+1
            else:
                break
        nums[i]=tmp
    head=nums[:k]
    for i in range((k-2)//2,-1,-1):#构建堆
        sift(head,i,k-1)
    for i in range(k,len(nums)):#对nums进行筛选
        if nums[i]>head[0]:
            head[0]=nums[i]
            sift(head,0,k-1)
    for i in range(k-1,-1,-1):#出数
        head[0],head[i]=head[i],head[0]
        sift(head,0,i-1)
    return head

#3.子集(非重复元素+重复元素)
def ziji11(nums:list):##重复元素
    res=[]
    nums.sort()
    def bt(start,path):
        res.append(path[:])
        for i in range(start,len(nums)):
            if start==i or nums[i]!=nums[i-1]:
                path.append(nums[i])
                bt(i+1,path)
                path.pop()
    bt(0,[])
    return res

#4.排列(非重复元素+重复元素)
def paile(nums):
    res=[]
    n=len(nums)
    def bt(path,nums):
        if len(path)==n:
            res.append(path[:])
        for i in range(len(nums)):
            if i==0 or nums[i]!=nums[i-1]:
                path.append(nums[i])
                bt(path,nums[:i]+nums[i+1:])
                path.pop()
    bt([],nums)
    return res

#5.分割回文字符串
def qiegea(s):
    res=[]
    def bt(start,path):
        if start>=len(s):
          res.append(path[:])
        for i in range(start,len(s)):
            p=s[start:i+1]
            if p==p[::-1]:
                path.append(p)
                bt(i+1,path)
                path.pop()
    bt(0,[])
    return res


def solve(board):
    n=len(board)
    m=len(board[0])
    def bt(i,j):
        if i<0 or i>=n or j<0 or j>=m or board[i][j]=='X':
            return
        if board[i][j]=='O':
            board[i][j]='#'
            bt(i+1,j)
            bt(i-1,j)
            bt(i,j+1)
            bt(i,j-1)
    for i in range(n):
        bt(i,0)
        bt(i,m-1)
    for j in range(1,m-1):
        bt(0,j)
        bt(n-1,j)
    for i in range(n):
        for j in range(m):
            if board[i][j]=='#':
                board[i][j]='O'
            else:
                board[i][j]='X'
    return board



def longestIncreasingPath(matrix):
    if not matrix or not matrix[0]:
        return 0
    row = len(matrix)
    col = len(matrix[0])
    lookup = [[0] * col for _ in range(row)]
    def dfs(i, j):
        if lookup[i][j] != 0:####有记录
            return lookup[i][j]
        # 方法一
        res = 1
        for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            tmp_i = x + i
            tmp_j = y + j
            if 0 <= tmp_i < row and 0 <= tmp_j < col and matrix[tmp_i][tmp_j] > matrix[i][j]:
                res = max(res, 1 + dfs(tmp_i, tmp_j))
        lookup[i][j] = max(res, lookup[i][j])
        return lookup[i][j]
    return max(dfs(i, j) for i in range(row) for j in range(col)),lookup
print(longestIncreasingPath(matrix = [[9,9,4],[6,6,8],[2,1,1]]))


def shoppingOffers(price,special, needs) -> int:
    memo = {}
    # 给定物品价格，大礼包分布和需要的物品数量，返回最少花费
    def shop(price, special, needs):
        if not any(needs):
            return 0
        if str((needs)) in memo:
            return memo[str(needs)]
        cost=0
        for i in range(len(needs)):
            cost+=(needs[i]*price[i])
        for sp in special:
            need=[]
            for i in range(len(needs)):
                if needs[i]<sp[i]:
                    break
                else:
                    need.append(needs[i]-sp[i])
            if len(needs)==len(need):
                cost=min(cost,sp[-1]+shop(price,special,need))
                memo[str(needs)]=cost
        return cost
    return shop(price, special, needs)
print(shoppingOffers(price = [2,5], special = [[3,0,5],[1,2,10]], needs = [3,2]))

def canPartitionKSubsets(nums,k):
    if k==1:
        return True
    t1,t2=sum(nums)//k,sum(nums)%k
    if t2!=0:
        return False
    nums.sort(reverse=True)
    if nums[-1]>t1:
        return False
    while nums and nums[0]==t1:
        nums.pop(0)
        k-=1
    if not nums:
        return True
    def bt(group,nums):
        if not nums:
            return True
        num=nums[0]
        for i in range(k):
            if group[i]+num<=t1:
                group[i]+=num
                if bt(group,nums[1:]):
                    return True
                group[i]-=num
            if group[i]==0:
                break
        return False
    return bt([0]*k,nums)












####前面第一个小于等于自己的元素
def fucntion1(nums:list):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res



print(fucntion1([3,1,3,4,61,1,7,12]))

####后面第一个小于自己的元素
def funtion2(nums:list):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res

####前面第一个大于等于自己的元素
def fucntion3(nums:list):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res


####后面第一个大于自己的元素
def funtion4(nums:list):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
























def rechazhao(nums,target):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<=end_zhen:
        mid=(end_zhen+start_zhen)//2
        if nums[mid]==target:
            return mid
        if nums[mid]<target:
            start_zhen=mid+1
        else:
            end_zhen=mid-1
    return start_zhen


def rechazhao1(nums,target):
    start_zhen=0
    end_zhen=len(nums)
    while start_zhen<end_zhen:
        mid=(end_zhen+start_zhen)//2
        if nums[mid]==target:
            return mid
        if nums[mid]<target:
            start_zhen=mid+1
        else:
            end_zhen=mid
    return -1


nums11=[1,2,5,7,9,10]

print(rechazhao(nums11,6))
print(rechazhao1(nums11,6))





