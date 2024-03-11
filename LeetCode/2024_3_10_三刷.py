# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/10 11:08
coding with comment！！！
"""

### 快排
def quick_sort(nums,left,right):
    def take_nums(nums,left,right):
        tmp = nums[left]
        while left < right:
            while left < right and nums[right] >= tmp:
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

# top_k

def top_k(nums,k):
    def sift(nums,low,high):
        tmp = nums[low]
        i = low
        j = 2 * i + 1
        while j <= high :
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
        sift(head,0,i)
    for i in range(k,len(nums)):
        if nums[i] > head[0]:
            head[0] = nums[i]
            sift(head,0,k - 1)
    for i in range(k-1,-1,-1):
        head[i],head[0] = head[0],head[i]
        sift(head,0,i - 1)
    return head


## rand5 - > rand 7
'''
5 * (rand5 - 1) + rand5
'''



### 回溯算法
## 子集

def subset(nums):
    res= []
    def bt(start,path):
        res.append(path[:])
        for i in range(start,len(nums)):
            if i == start or nums[i] != nums[i - 1]:
                path.append(nums[i])
                bt(i + 1,path)
                path.pop()
    bt(0,[])
    return res


# pailie
def paile(nums):
    res = []
    n = len(nums)
    nums.sort()
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

def qiege_(s):
    res = []
    def bt(start,path):
        if start >= len(s):
            res.append(path[:])
        for i in range(start,len(s)):
            p = s[start:i + 1]
            if p == p[::-1]:
                path.append(p)
                bt(i + 1,path)
                path.pop()
            else:
                continue
    bt(0,[])
    return res


## 二分查找

def binary(nums,target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return mid
        elif nums[start] > target:
            end = mid - 1
        else:
            start = mid + 1
    return start
### 请注意 start 是需要插入的位置。

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
    return end


def solve(nums: str) -> int:
    if len(nums) == 1:
        if nums[0] != "0":
            return 1
        else:
            return 0
    else:
        dp = [0] * len(nums)  ## dp[i] 表示nums[0-i]能组成的数量
        if nums[0] != "0":
            dp[0] = 1
        s = nums[0:2]
        if len(s) == len(str(int(s))):
            if int(s) <= 26 and int(s) not in [10, 20]:
                dp[1] = 2
            elif int(s) <= 26 and int(s) in [10, 20]:
                dp[1] = 1
            elif int(s) > 26 and s[-1] != "0":
                dp[1] = 1
            elif int(s) > 26 and s[-1] == "0":
                dp[1] = 0
        else:
            dp[1] = 0

        for i in range(2, len(nums)):
            if nums[i] != "0":
                dp[i] += dp[i - 1]

            if len(nums[i - 1:i + 1]) == len(str(int(nums[i - 1:i + 1]))) and int(nums[i - 1:i + 1]) <= 26:
                dp[i] += dp[i - 2]

        return dp[-1]



def get_Listnode_lens(head):
    if not head:
        return 0
    elif not head.next:
        return 1
    else:
        ans = 0
        while head:
            ans += 1
            head = head.next
        return ans


class Node():
    def __init__(self,val):
        self.val = val
        self.next = None


a = Node(1)
b = Node(2)
c = Node(3)

a.next = b
b.next = c

print(get_Listnode_lens(a))



nums = [1,2,43]
nums.sort(key = lambda )









