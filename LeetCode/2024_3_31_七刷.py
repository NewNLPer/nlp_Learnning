# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/31 12:04
coding with comment！！！
"""
## 快排
def qiuck_sort(nums,left,right):
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
        qiuck_sort(nums,left,mid - 1)
        qiuck_sort(nums,mid + 1,right)
        return nums

def tok_k(nums,k):
    def sift(nums,low,high):
        tmp = nums[low]
        i = low
        j = 2 * i + 1
        while j <= high:
            if j + 1 <= high and nums[j+1] < nums[j]:
                j += 1
            if nums[j] < tmp:
                nums[i] = nums[j]
                i = j
                j = 2 * i + 1
            else:
                break
        nums[i] = tmp
    head = nums[:k]
    for i in range((k-2)//2,-1,-1):
        sift(head,i,k-1)
    for i in range(k,len(nums)):
        if nums[i] > head[0]:
            head[0] = nums[i]
            sift(head,0,k-1)
    for i in range(k-1,-1,-1):
        head[0],head[i] = head[i],head[0]
        sift(head,0,i-1)
    return head

nums = [1,2,4,5,3,542,2,54]
print(tok_k(nums,3))

"""
堆排序的时间复杂度
k+nlog(k)
"""

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
    return -1

"""
1.有序
2.start的位置就是需要插入的位置
"""

## 找最右

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





