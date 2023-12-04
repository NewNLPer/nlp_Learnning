# -*- coding:utf-8 -*-
# @Time      :2022/9/16 9:57
# @Author    :Riemanner
# Write code with comments !!!
def function(n):
    return (n*(n+1))//2
def numSubarraysWithSum(nums,goal):
    if goal!=0:
        sum0=[0]*len(nums)
        sum0[0]=nums[0]
        dic_c={}
        res=0
        for i in range(len(nums)):
            sum0[i]=sum0[i-1]+nums[i]
            dic_c[sum0[i]]=dic_c.get(sum0[i],0)+1
        for i in range(len(sum0)):
            if sum0[i]==goal:
                res+=1
            res+=dic_c.get(sum0[i]+goal,0)
        return res
    else:
        nums=nums+[1]
        start_zhen=0
        end_zhen=0
        res=0

        while end_zhen<len(nums):
            while start_zhen<len(nums):
                if nums[start_zhen]==1:
                    start_zhen+=1
                else:
                    break
            end_zhen=start_zhen+1
            while end_zhen<len(nums):
                if nums[end_zhen]==0:
                    end_zhen+=1
                else:
                    res+=function(end_zhen-start_zhen)
                    start_zhen=end_zhen+1
                    end_zhen+=1
                    break
        return res
print(numSubarraysWithSum(nums = [0,1,0,1,0], goal = 0))