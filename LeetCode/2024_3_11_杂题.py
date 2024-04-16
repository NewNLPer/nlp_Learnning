# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/3/11 18:01
coding with comment！！！
"""
## 快速排序
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

## top-k
def top_k(nums,k):
    def sift(nums,low,high):
        tmp = nums[low]
        i = low
        j = 2 * i + 1
        while j <= high:
            if j + 1 <= high and nums[j + 1] < nums[j]:
                j += 1
            if nums[j] < tmp :
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
            sift(head,0,k - 1)
    for i in range(k - 1,-1,-1):
        head[0],head[i] = head[i],head[0]
        sift(head,0,i - 1)
    return head



### rand5 -> rand7
'''
5 * (rand5 - 1) = [0,5,10,15,20]
5 * (rand5 - 1) + rand5 = [1,25]
'''
nums = [1,24]
nums.sort(key=)