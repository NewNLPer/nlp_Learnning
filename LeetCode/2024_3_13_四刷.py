# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/13 19:03
coding with comment！！！
"""
#### 快排
def quick_sort(nums,left,right):
    def take_nums(nums,left,right):
        tmp = nums[left]
        while left < right:
            while left < right and nums[right] >= tmp:
                right -= 1
            nums[left],nums[right] = nums[right],nums[left]
            while left < right and nums[left] <= tmp:
                left += 1
            nums[left] ,nums[right] = nums[right] ,nums[left]
        nums[left] = tmp
        return left
    if left < right:
        mid = take_nums(nums,left,right)
        quick_sort(nums,left,mid - 1)
        quick_sort(nums,mid + 1,right)
        return nums

### top_k
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
    for i in range((k - 2) // 2, -1,-1):
        sift(head,0,i)
    for i in range(k,len(nums)):
        if nums[i] > head[0]:
            head[0] = nums[i]
            sift(head,0,k - 1)
    for i in range(k - 1 ,-1,-1):
        head[0],head[i] = head[i] ,head[0]
        sift(head,0,i - 1)
    return head

### rand5 -> rand7
"""
5 * (rand5 - 1) [0,5,10,15,20]
5 * (rand5 - 1) + rand5 = [1,25]
"""

### erfnchazhao

def erfenchaozhao(nums,target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    return start

def get_r(nums,target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    if end < 0 or nums[end] != target:
        return -1
    return  end


### 子集
def subset(nums):
    res = []
    nums.sort()
    def bt(start,path):
        res.append(path[:])
        for i in range(start,len(nums)):
            if i == start or nums[i] != nums[i - 1]:
                path.append(nums[i])
                bt(i + 1,path)
                path.pop()
    bt(0,[])
    return res


### 排列
def pailie(nums):
    res = []
    n = len(nums)
    nums.sort()
    def bt(nums,path):
        if len(path) == n:
            res.append(path[:])
            return
        for i in range(len(nums)):
            if not i or nums[i] != nums[i - 1]:
                path.append(nums[i])
                bt(nums[:i]+nums[i+1:],path)
                path.pop()
    bt(nums,[])
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





















