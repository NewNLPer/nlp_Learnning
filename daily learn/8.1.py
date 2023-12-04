# -*- coding:utf-8 -*-
# @Time      :2022/8/1 10:08
# @Author    :Riemanner
# Write code with comments !!!
def generateTheString(n):
    res=['a','b']
    if n%2==0:
        return res[0]*(n-1)+res[1]
    else:
        return res[0]*n

def canSeePersonsCount(heights):
    n = len(heights)
    ans = [0] * n
    s = list()
    for i in range(n - 1, -1, -1):###倒着维护一个单调递减栈
        while s:
            ans[i] += 1
            if heights[i] > heights[s[-1]]:
                s.pop()
            else:
                break
        s.append(i)
    return ans





def largestRectangleArea(heights) -> int:
