# -*- coding:utf-8 -*-
# @Time      :2022/7/26 9:29
# @Author    :Riemanner
# Write code with comments !!!
###单调栈
def dandaio_stack(arr:list):####右边比自己大的第一个元素,维护一个单调递减栈
    res=[-1]*len(arr)
    stack=[]
    for i in range(len(arr)-1,-1,-1):
        while stack and stack[-1]>arr[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(arr[i])
    for j in range(len(res)):
        if res[j]!=-1:
            res[j]=arr[j]-res[j]
        else:
            res[j]=arr[j]
    return res
print(dandaio_stack([10,1,1,6]))


# def dandiao_Zhan(arr:list):#####右边比自己小的第一个元素，维护一个单调递增栈
#     res=[-1]*len(arr)
#     stack=[]
#     for i in range(len(arr)-1,-1,-1):
#         while stack and stack[-1]>=arr[i]:
#             stack.pop()
#         if stack:
#             res[i]=stack[-1]
#         stack.append(arr[i])
#     return res
# print(dandiao_Zhan([1,3,4,5,2,9,6]))


def nextGreaterElement(nums1,nums2):
    res ={}
    res1=[]
    stack = []
    for i in range(len(nums2) - 1, -1, -1):
        while stack and stack[-1] <= nums2[i]:
            stack.pop()
        if stack:
            res[nums2[i]] = stack[-1]
        else:
            res[nums2[i]]=-1
        stack.append(nums2[i])
    for j in nums1:
        res1.append(res[j])
    return res1


def nextGreaterElements(nums):
    arr=nums+nums
    res=[-1]*len(arr)
    stack=[]
    for i in range(len(arr)-1,-1,-1):
        while stack and stack[-1]<=arr[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(arr[i])
    return res[:len(nums)]

expression = "1/2+1/10-1/3"
expression=expression.replace('+','/')
expression=expression.replace('-','/-')
# print(expression)
# expression=expression.split('/')
# print(expression)
import math
def gcd_many(s):
    g = 0
    for i in range(len(s)):
        if i == 0:
            g = s[i]
        else:
            g=math.gcd(g,s[i])

    return g
import math

def lcm(a, b):
    p = 1
    i = 2
    while i <= min(a, b):
        if a % i == 0 and b % i == 0:
            p *= i
            a, b = a // i, b // i
        else:
            i += 1
    p = p * a * b
    return p
def fractionAddition(expression:str):
    expression = expression.replace('+', '/')
    expression = expression.replace('-', '/-')
    expression = expression.split('/')
    if expression[0]=='':
        expression.pop(0)
    for i in range(1,len(expression),2):
        if i==1:
            g=int(expression[i])
        else:
            g=lcm(g,int(expression[i]))
    res=0
    for j in range(0,len(expression)-1,2):
        res+=(int(expression[j])*(g//int(expression[j+1])))
    s=math.gcd(res,g)
    return str(res//s)+'/'+str(g//s)



####下一个比我的xiao的元素
def dan_diao(nums):
    stack=[]
    res=[0]*len(nums)
    for i in nums:
        while stack and stack[-1]>i:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res









def largestRectangleArea(heights):
    stack = []
    heights = [0] + heights + [0]
    res = 0
    for i in range(len(heights)):
        #print(stack)
        while stack and heights[stack[-1]] > heights[i]:
            tmp = stack.pop()
            res = max(res, (i - stack[-1] - 1) * heights[tmp])
        stack.append(i)
    return res



def removeKdigits(num: str, k: int) -> str:
    stack = []
    for d in num:
        while stack and k and stack[-1] > d:
            stack.pop()
            k -= 1
        stack.append(d)
    if k > 0:
        stack = stack[:-k]
    return ''.join(stack).lstrip('0') or "0"


def dailyTemperatures(temperatures):###维护一个单调递增站
    nums=temperatures
    stack=[]
    res=[0]*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]-i
        stack.append(i)
    return res

def dailyTemperatures2(temperatures):####维护一个单调递减栈
    nums=temperatures
    stack=[]
    res=[0]*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<=nums[i]:
            res[stack[-1]] = i-stack[-1]
            stack.pop()
        stack.append(i)
    return res


def removeKdigits222(num,k):
    stack = []
    for d in num:
        while stack and k and stack[-1] > d:
            stack.pop()
            k -= 1
        stack.append(d)
    if k > 0:
        stack = stack[:-k]
    return ''.join(stack).lstrip('0') or "0"




