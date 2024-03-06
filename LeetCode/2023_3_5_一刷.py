# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/5 12:00
coding with comment！！！
"""
## 1.快速排序
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
        take_nums(nums,left,mid - 1)
        take_nums(nums,mid + 1,right)
        return nums
nums = [1,3,2,2,1,3,4,6]
# print(quick_sort(nums,0,len(nums) - 1))

## 2.top_k 问题
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
    for i in range(k-1,-1,-1):
        head[0],head[i] = head[i],head[0]
        sift(head,0,i-1)
    return head
# print(top_k(nums,5))

## 3 rand5 去实现rang7
'''
首先一定要产生0，然后再相加
'''
def lengthOfLongestSubstring(s):
    if len(s) <= 1:
        return len(s)
    else:
        start_zhen = 0
        end_zhen = 1
        res = 1
        dic = {}
        dic[s[start_zhen]] = 1
        dic[s[end_zhen]] = dic.get(s[end_zhen],0) + 1
        while end_zhen < len(s):
            if len(dic) == sum(dic.values()):
                res = max(res,end_zhen - start_zhen + 1)
                end_zhen += 1
                if end_zhen < len(s):
                    dic[s[end_zhen]] = dic.get(s[end_zhen],0) + 1
            else:
                pass

## 接雨水 - 画图示意

## 二分查找
 # 左闭右闭
def erfenchazhao(nums,target): # 如果找不到，start的地方就是需要插的位置
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
    return -1,start

nums = [1,4,5,6,7,9]
# print(erfenchazhao(nums,6.5))
### 按照排序，其中存在重复元素，要找到

def get_r(nums,target):
    start_zhen = 0
    end_zhen = len(nums)-1
    while start_zhen <= end_zhen:
        mid = (start_zhen+end_zhen)//2
        if nums[mid] > target:
            end_zhen = mid - 1
        elif nums[mid] < target:
            start_zhen = mid + 1
        else:
            start_zhen = mid + 1
    if end_zhen < 0 or nums[end_zhen] != target:
        return -1
    return end_zhen

nums = [1,2,3,4,5]

# print(get_r(nums,3))



def search(nums, target):
    if len(nums)==1:
        if target in nums:
            return 0
        else:
            return -1
    else:
        start=0
        end=len(nums)-1
        while start<=end:
            mid=(start+end)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]>=nums[start]:#左侧有序
                if nums[start]<=target<nums[mid]:
                    end=mid-1
                else:
                    start=mid+1
            else:
                if nums[mid]<target<=nums[end]:
                    start=mid+1
                else:
                    end=mid-1
        return -1


## 回溯算法

def subset(nums):
    res = []
    nums.sort()
    def bt(path,start):
        res.append(path[:])
        for i in range(start,len(nums)):
            path.append(nums[i])
            bt(path,i+1)
            path.pop()
    bt([],0)
    return res
print(subset([1,2,3]))

