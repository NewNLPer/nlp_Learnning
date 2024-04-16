# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/9 10:14
coding with comment！！！
"""

# 一些杂题
#  快排
def quick_sort(nums,left,right):
    def take_nums(nums,left,right):
        tmp = nums[left]
        while left < right:
            while left < right and nums[right] >=tmp:
                right -= 1
            nums[left],nums[right] = nums[right],nums[left]
            while left < right and nums[left] <= tmp:
                left += 1
            nums[left],nums[right] = nums[right],nums[left]
        nums[left] = tmp
        return left
    if left < right:
        mid = take_nums(nums,left,right)
        quick_sort(nums,left,mid - 1)
        quick_sort(nums,mid + 1,right)
        return nums

# tok_P
def top_k(nums,k):
    def sift(nums,low,high):
        tmp = nums[low]
        i = low
        j = 2 * i + 1
        while j <= high:
            if j + 1 <= high and nums[j + 1] < nums[j]:
                j += 1
            if nums[j] < tmp:
                nums[i] = nums[j]
                i = j
                j = 2 * i + 1
            else:
                break
        nums[i] = tmp
    head = nums[:k]
    for i in range((k - 2) // 2,-1,-1):
        sift(head,i,k - 1)
    for i in range(k,len(nums)):
        if nums[i] > head[0]:
            head[0] = nums[i]
            sift(head,0,k-1)
    for i in range(k - 1,-1,-1):
        head[0],head[i] = head[i],head[0]
        sift(head,0,i - 1)
    return head

nums = [1,3,4,23,2,5,3]

## rand5 -> rand7
'''
5 * (rand5 - 1) = [0,5,10,15,20]
5 * (rand5 - 1) + rand5 = [1 - 25]
看看mod 

'''

## 接雨水 暂时不说了，有张图就可以说明问题

# 二分查找


def erfenchazhao(nums,target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            end = mid -1
        else:
            start = mid + 1
    return (-1,start)

'''
请注意start也是如果没有，则需要插入的位置
'''

## 重复元素找最右边
def get_r(nums,target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end)
        if nums[mid] > target:
            end = mid -1
        else:
            start = mid + 1
    if end < 0 or nums[end] != target:
        return -1
    return  end



 ## 回溯算法，
  ## 子集
def subset(nums):
    res = []
    def bt(path,start):
        res.append(path[:])
        for i in range(start,len(nums)):
            if i == start or nums[i] != nums[i - 1]:
                path.append(nums[i])
                bt(path,i+1)
                path.pop()
    bt([],0)
    return res


## 排列
def pailie(nums):
    res = []
    nums.sort()
    n = len(nums)
    def bt(path,nums):
        if len(path) == n:
            res.append(path[:])
        for i in range(len(nums)):
            if not i or nums[i] != nums[i - 1]:
                path.append(nums[i])
                bt(path,nums[:i] + nums[i + 1:])
                path.pop()
    bt([],nums)
    return res

def qiege(s):
    res = []
    def bt(start,path):
        if start >= len(s):
            res.append(path[:])
        for i in range(start,len(s)):
            p = s[start:i + 1]
            if p == p[::-1]:
                path.append(p)
                bt(i+1,path)
                path.pop()
            else:
                continue
    bt(0,[])
    return res


def getPermutation(n: int, k: int) -> str:
    res = []
    nums = list(range(1,n+1))
    def bt(path,nums):
        if len(path) >= n:
            res.append(''.join(path))
        for i in range(len(nums)):
            path.append(str(nums[i]))
            bt(path,nums[:i]+nums[i+1:])
            path.pop()
    bt([],nums)
    return res[k-1]
print(getPermutation(9,136371))



### attention的计



import torch
import torch.nn as nn


class Muti_Head_Attention(nn.Module):
    def __init__(self):
        super(Muti_Head_Attention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_k * n_head)


    def get_atten_score(self,Q,K,V,mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

    def forward(self,x_input):
        batch_size = x_input.size()[0]
        Q = self.W_Q(x_input).view(batch_size,-1,n_head,d_k)
        K = self.W_V(x_input).view(batch_size,-1,n_head,d_k)
        V = self.W_V(x_input).view(batch_size,-1,n_head,d_k)
        mask

        context = self.get_atten_score(Q,K,V,mask)
        return context













































































