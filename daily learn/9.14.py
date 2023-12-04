# -*- coding:utf-8 -*-
# @Time      :2022/9/14 9:15
# @Author    :Riemanner
# Write code with comments !!!
def funtion1(nums,target):###左闭右闭
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
    return -1

def funtion2(nums,target):###左闭右开
    start_zhen=0
    end_zhen=len(nums)
    while start_zhen<end_zhen:
        mid=(start_zhen+end_zhen)//2
        if nums[mid]>target:
            end_zhen=mid
        elif nums[mid]<target:
            start_zhen=mid+1
        else:
            return mid
    return -1

def longestOnes(nums,k):
    start_zhen=0
    end_zhem=0
    count0=0
    res=0
    while end_zhem<len(nums):
        if nums[end_zhem]==0:
            count0+=1
            if count0<=k:
                end_zhem+=1
        else:
            end_zhem+=1
        if count0>k:
            res = max(res, end_zhem - start_zhen )
            while start_zhen<=end_zhem:
                if nums[start_zhen]==1:
                    start_zhen+=1
                else:
                    count0-=1
                    start_zhen+=1
                    end_zhem+=1
                    break
    res = max(res, end_zhem - start_zhen)
    return res
print(longestOnes([1,0,0,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1],9))
