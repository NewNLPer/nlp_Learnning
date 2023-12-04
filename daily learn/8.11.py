# -*- coding:utf-8 -*-
# @Time      :2022/8/11 14:19
# @Author    :Riemanner
# Write code with comments !!!
####单调栈的复习
####寻找下一个比自己大的元素,维护一个单调递减栈
###倒着遍历
def funtion1(nums):
    stack=[]
    res=[0]*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res

###正着遍历
def function2(nums):
    res=[0]*len(nums)
    stack=[]
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<=nums[i]:
            res[stack[-1]]=i
            stack.pop()
        stack.append(i)
    return res

def groupThePeople(groupSizes):
    dic_c={}
    res=[]
    for i in range(len(groupSizes)):
        dic_c[groupSizes[i]]=dic_c.get(groupSizes[i],[])+[i]
    for key in dic_c:
        s=len(dic_c[key])//key
        start_zhen=0
        while s!=0:
            res1=[]
            while len(res1)<key:
                res1.append(dic_c[key][start_zhen])
                start_zhen+=1
            s-=1
            res.append(res1)
    return res


print(groupThePeople(groupSizes = [2,1,3,3,3,2]))


