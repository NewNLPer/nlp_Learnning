# -*- coding:utf-8 -*-
# @Time      :2022/8/10 10:57
# @Author    :Riemanner
# Write code with comments !!!

def solveEquation(equation: str) -> str:
    start_zhen=0
    while equation[start_zhen]!='=':
        start_zhen+=1
    s1=equation[:start_zhen].replace('-','+-').split('+')
    s2=equation[start_zhen+1:].replace('-','+-').split('+')
    zuo_x=0
    zuo_shu=0
    you_x=0
    you_shu=0
    for item in s1:
        if item=='':
            continue
        elif item[-1]!='x' and item[-1]!='-x':
            zuo_shu+=int(item)
        else:
            if item=='x':
                zuo_x+=1
            elif item=='-x':
                item-=1
            else:
                zuo_x+=int(item[:len(item)-1])
    for item in s2:
        if item=='':
            continue
        elif item[-1]!='x' and item[-1]!='-x':
            you_shu+=int(item)
        else:
            if item=='x':
                you_x+=1
            elif item=='-x':
                you_x-=1
            else:
                you_x+=int(item[:len(item)-1])
    zong_x=zuo_x-you_x
    zong_shu=you_shu-zuo_shu
    ###无解
    if zong_x==0 and zong_shu!=0:
        return 'No solution'
    ###无限解
    elif zong_x==0 and zong_shu==0:
        return 'Infinite solutions'
    ###唯一解
    else:
        return 'x=%d'%(zong_shu//zong_x)






def subSort(array):
    if not array: return [-1,-1]
    m = n = -1
    left = array[-1]
    for i in range(len(array)-2,-1,-1):
        if array[i] > left:
            m = i
        else:
            left = array[i]

    right = array[0]
    for i in range(1,len(array)):
        if array[i] < right:
            n = i
        else:
            right = array[i]
    return [m,n]


def asteroidCollision(asteroids):
    stack=[]
    for i in range(len(asteroids)):
        b=0
        while stack and stack[-1]>0 and asteroids[i]<0:
            if stack[-1]==abs(asteroids[i]):
                stack.pop()
                b=1
                break
            elif stack[-1]<abs(asteroids[i]):
                stack.pop()
                b=0
            else:
                b=1
                break
        if not b:
            stack.append(asteroids[i])
    return stack

def reformat(s):
    res1=[]###数字
    res2=[]###字母
    res=''
    for i in range(len(s)):
        if s[i].isdigit():
            res1.append(s[i])
        else:
            res2.append(s[i])
    if len(res1)==len(res2):
        for i in range(len(res1)):
            res+=res1[i]
            res+=res2[i]
        return res
    elif abs(len(res2)-len(res1))==1:
        if len(res2)>len(res1):
            for i in range(len(res1)):
                res+=res2[i]
                res+=res1[i]
            res+=res2[-1]
            return res
        else:
            for i in range(len(res2)):
                res+=res1[i]
                res+=res2[i]
            res+=res1[-1]
            return res
    else:
        return ''

import collections
def maxSlidingWindow(nums,k):
    res = []
    queue = collections.deque()
    for i, num in enumerate(nums):
        if queue and i-queue[0]==k:##说明维护的队列的第一个元素不在目前的滑动窗口内
            queue.popleft()
        while queue and nums[queue[-1]]<num:
            queue.pop()
        queue.append(i)
        if i>=k-1:
            res.append(nums[queue[0]])
    return res
print(maxSlidingWindow(nums = [1,3,-1,-3,5,3,6,7], k = 3))