# -*- coding:utf-8 -*-
# @Time      :2022/7/28 8:09
# @Author    :Riemanner
# Write code with comments !!!
def dailyTemperatures(temperatures):###维护一个单调递减栈，寻找后面比我大的元素
    nums=temperatures
    stack=[]
    res=[0]*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res



def dailyTemperatures2(temperatures):####维护一个单调递减栈，存栈(深刻理解，次站存的是当前元素之前的元素)
    nums=temperatures
    stack=[]
    res=[0]*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<=nums[i]:
            res[stack[-1]] = i
            stack.pop()
        stack.append(i)
    return res


####开始默写两种单调栈的写法
###从前到后，下一个，也就是右边
def function1(nums):
    res=[0]*len(nums)
    stack=[]
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<=nums[i]:
            res[stack[-1]]=i
            stack.pop()
        stack.append(i)
    return res

####从后到前，下一个也就是右边
def function2(nums):
    res=[0]*len(nums)
    stack=[]
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res

def function3(nums):###左边第一个比自己小的
    res=['*']*len(nums)
    stack=[]
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res

def function4(nums):###左边第一个比自己大的
    res=['*']*len(nums)
    stack=[]
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res


def arrayRankTransform(arr):
    res=arr[:]
    res=list(set(res))
    res.sort()
    dic_c={}
    for i in range(len(res)):
        dic_c[res[i]]=i+1
    path=['*']*len(arr)
    for i in range(len(arr)):
        path[i]=dic_c[arr[i]]
    return path


def maxChunksToSorted(arr):
    result = 0
    for i in range(len(arr)):
        if max(arr[:i+1]) == i:
            result += 1
    return result


def maxChunksToSorted2(arr):
    leftMax,rightMin=[0]*len(arr),[0]*len(arr)
    leftMax[0]=arr[0]
    for i in range(1,len(arr)):###leftmax[i]表示0-i的最大值
        leftMax[i]=max(leftMax[i-1],arr[i])
    rightMin[len(arr)-1]=arr[len(arr)-1]
    for i in range(len(arr)-2,-1,-1):###rightmin表示末尾带i的最小值
        rightMin[i]=min(rightMin[i+1],arr[i])
    ans=1
    for i in range(1,len(arr)):
        ans+=rightMin[i]>=leftMax[i-1]
    return ans



def function33(nums):###左边第一个比自己小的
    res=['*']*len(nums)
    stack=[]
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    res1=0
    for j in range(len(res)):
        if res[j]!='*':
            res2=res[j]
            while res[res2]!='*':
                res3=res[res2]
                res2=res3
            res1=max(res1,j-res2)
    return res1



def smallestSubsequence(s):
    stack=[]
    for i in range(len(s)):
        if s[i] in stack:
            continue
        while stack and stack[-1]>s[i] and stack[-1] in s[i+1:]:
            stack.pop()
        stack.append(s[i])
    return ''.join(stack)
'''
基础知识1： 设nums[j],nums[k]为nums[i]左右最近的比nums[i]小的值
            则以nums[i]为最小值的子数组共有(i - j) * (k - i)个 （不超过j,k范围且包含i的子数组总数）
            最大值也同理

基础知识2： 子数组的范围和可以表示为所有子数组最大值之和 减去 子数组最小值之和（简单数学推导）

由1、2可知：如果能创建两个数组minLeft和minRight，下标i的值保存基础知识1中j和k的下标
            则(i - minLeft[i]) * (minRight[i] - j)即为以nums[i]为最小值的子数组总数
            用此值再乘以 nums[i] 即可得到nums[i]为最小值的子数组的最小值之和
            对每一个 0 < i < len - 1 做此操作并叠加最小值之和，即可得到整个数组的子数组最小值之和B 
            最大值同理，由基础知识2即可求得子数组范围和

因此，本题的难点在于：如何构造minLeft和minRight数组(maxLeft, maxRight)

以minLeft为例，利用单调栈构造数组，具体如下：
1. 从左到右遍历nums数组
2. 对于遍历到的nums[i]，执行出栈(pop)直到：
    1) 栈空
    2) 栈顶(top)小于nums[i]
此时，可记录minLeft[i]的值为栈顶元素 或 -1 (栈空时)，然后将i压入栈中(push)

经此操作，可以构造出minLeft数组；类似也可构造maxLeft,minRight,maxRight

为保证最大值/最小值的唯一性，在出现相同的值时，以下标的大小作为第二判断标准，下标小的认为是更小的值

'''
###下一个更小的元素
def function11(nums):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
###前一个更小的
def function111(nums):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
###下一个更大的
def function22(nums):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
###前一个更大的
def function222(nums):
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]<=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res

def subArrayRanges(nums):
    nums1=[0]+nums+[0]
    nums2=[9999999]+nums+[9999999]
    res1=function111(nums1)###前一个更小
    res2=function11(nums1)###下一个更小
    res3=function222(nums2)###前一个更大
    res4=function22(nums2)###下一个更大
    res5=0
    for i in range(1,len(nums1)-1):
        if res1[i]!='*':
            res5+=((i-res1[i])*(res2[i]-i)*nums[i-1])
    return res5
print(subArrayRanges([4,-2,-3,4,1]))