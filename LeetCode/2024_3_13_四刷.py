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




def get_corre(dic_1,dic_2):
    for key in dic_1:
        nums = dic_2.get(key,-1)
        if nums < dic_1[key]:
            return False
    return True

def minWindow(s: str, t: str) -> str:
    if len(s) == len(t):
        if s!= t:
            return ""
        else:
            return s
    elif len(s) < len(t):
        return ""
    else:
        res = ["",99999]
        dic_1 = {} # 为基准
        for i in range(len(t)):
            dic_1[t[i]] = dic_1.get(t[i],0) + 1
        dic_2 = {}
        start = 0
        end = 1
        dic_2[s[start]] = 1
        dic_2[s[end]] = dic_2.get(s[end],0) + 1
        while end < len(s):
            if not get_corre(dic_1,dic_2):
                end += 1
                if end < len(s):
                    dic_2[s[end]] = dic_2.get(s[end],0) + 1
            else:
                if end - start + 1 < res[-1]:
                    res = [s[start:end + 1],end - start + 1]
                dic_2[s[start]] -= 1
                if not dic_2[s[start]]:
                    del dic_2[s[start]]
                start += 1
        return res[0]



import re
def decodeString(s:str)->str:
    patter=re.compile(r"(\d+)\[(\w+)\]")
    m=patter.findall(s)
    print(m)
    while m:
        for num,char in m:
            s=s.replace("%s[%s]" % (num,char),char*int(num))
        m=patter.findall(s)
        print(m)
    return s

print(decodeString("abc3[cd]xyz"))











