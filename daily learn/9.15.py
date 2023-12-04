# -*- coding:utf-8 -*-
# @Time      :2022/9/15 15:27
# @Author    :Riemanner
# Write code with comments !!!
def numSubarrayProductLessThanK(nums: list[int], k: int) -> int:
    res=0
    for i in range(len(nums)):
        if nums[i]<k:
            res+=1
    sum0=[0]*len(nums)
    sum0[0]=nums[0]
    for i in range(1,len(nums)):
        sum0[i]=sum0[i-1]*nums[i]
    sum0=[1]+sum0
    start_zhen=1
    end_zhen=2
    while end_zhen<len(sum0):
        if sum0[end_zhen]//sum0[start_zhen-1]<k:
            res+=1
            end_zhen+=1
        else:
            start_zhen+=1
            if start_zhen==end_zhen:
                end_zhen=start_zhen+1
        if end_zhen==len(sum0):
            start_zhen+=1
            while start_zhen < len(sum0) - 1:
                if sum0[-1]//sum0[start_zhen-1]<k:
                    res+=1
                    start_zhen+=1
                else:
                    start_zhen+=1
    return res


def minSubArrayLen(target,nums):
    for i in range(len(nums)):
        if nums[i]>=target:
            return 1
    sum0=[0]*len(nums)
    sum0[0]=nums[0]
    for i in range(1,len(nums)):
        sum0[i]=sum0[i-1]+nums[i]
    sum0=[0]+sum0
    start_zhen=1
    end_zhen=2
    res=float('inf')
    while end_zhen<len(sum0):
        if sum0[end_zhen]-sum0[start_zhen-1]<target:
            end_zhen+=1
        else:
            res=min(res,end_zhen-start_zhen+1)
            start_zhen+=1
            if start_zhen==end_zhen:
                end_zhen=start_zhen+1
        if end_zhen==len(sum0):
            start_zhen+=1
            while start_zhen < len(sum0) - 1:
                if sum0[-1]-sum0[start_zhen-1]>=target:
                    res=min(res,end_zhen-start_zhen)
                    start_zhen+=1
                else:
                    break
    if res==float('inf'):
        return 0
    else:
        return res
print(minSubArrayLen(target = 11, nums = [1,1,1,1,1,1,1,1]))