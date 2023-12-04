# -*- coding:utf-8 -*-
# @Time      :2022/8/3 15:18
# @Author    :Riemanner
# Write code with comments !!!
def orderlyQueue(s,k):
    if k==1:
        p=len(s)
        res=s
        while p!=0:
            s=s[1:]+s[0]
            res=min(res,s)
            p-=1
        return res
    else:
        list1=[]
        list1+=s
        list1.sort()
        return ''.join(list1)
def minSubsequence(nums: list[int]):
    nums.sort(reverse=True)
    res=[0]*len(nums)
    for i in range(len(nums)):
        if i==0:
            res[i]=nums[i]
        else:
            res[i]=nums[i]+res[i-1]
    for i in range(len(nums)):
        if res[i]>res[-1]-res[i]:
            return nums[:i+1]
from sortedcontainers import SortedList
def lexicalOrder(n):
    s=SortedList([],key=lambda x:str(x))
    for i in range(1,n+1):
        s.add(i)
    return list(s)

def stringMatching(words):
    res=[]
    words.sort(key=lambda x:len(x))
    for i in range(len(words)-1):
        for j in range(i+1,len(words)):
            if words[i] in words[j]:
                res.append(words[i])
                break
    return res



def fucntion1(nums:list):
    nums=[0]+nums+[0]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
####后面第一个小于自己的元素
def funtion2(nums:list):
    nums=[0]+nums+[0]
    stack=[]
    res=['*']*len(nums)
    for i in range(len(nums)-1,-1,-1):
        while stack and nums[stack[-1]]>=nums[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        stack.append(i)
    return res
def validSubarraySize(nums,threshold):
    res1=fucntion1(nums)####前小
    res2=funtion2(nums)###后大
    for i in range(1,len(res1)):
        if nums[i-1]>threshold/(res2[i]-res1[i]-1):
            return res2[i]-res1[i]-1
    return -1

from collections import Counter
def reorganizeString(S: str) -> str:
    l = len(S)
    if l < 2:
        return S
    c = Counter(S)
    m = c.most_common()
    if m[0][1] > (l + 1) // 2:
        return ''
    ret = [' '] * l
    i = 0
    for mm in m:
        for j in range(mm[1]):
            ret[i] = mm[0]
            i += 2
            if i >= l:
                i = 1
    return ''.join(ret)


def validPalindrome(s: str) -> bool:
    start_zhen=0
    end_zhen=len(s)-1
    while start_zhen<=end_zhen:
        if s[start_zhen]==s[end_zhen]:
            start_zhen+=1
            end_zhen-=1
        else:
            if s[start_zhen+1:end_zhen+1]==s[start_zhen+1:end_zhen+1][::-1] or s[start_zhen:end_zhen]==s[start_zhen:end_zhen][::-1]:
                return True
            else:
                return False
    return True

def funtion(nums):
    res=[]
    for i in range(len(nums)):
        if type(nums[i])==int:
            res.append(nums[i])
        else:
            item=str(nums[i])
            start_zhen=0
            while start_zhen<=len(item)-1:
                if item[start_zhen].isdigit() or item[start_zhen]=='-':
                    end_zhen=start_zhen+1
                    while end_zhen<=len(item)-1 and item[end_zhen].isdigit():
                        end_zhen+=1
                    res.append(int(item[start_zhen:end_zhen]))
                    start_zhen=end_zhen
                else:
                    start_zhen+=1
    return res



def arithmeticTriplets(nums,diff):
    res=0
    for i in range(len(nums)):
        item=nums[i]
        c=1
        while item+diff in set(nums):
            item+=diff
            c+=1
        if c>=3:
            res+=1
    return res

def longestIdealString(s: str, k: int):
    f = [0] * 26
    for c in s:
        c = ord(c) - ord('a')####将字母转换成0-25数字
        f[c] = 1 + max(f[max(c - k, 0): c + k + 1])###范围内寻找
    return max(f)


def validPartition(nums):
    if len(nums)>=3:
        dp=[False]*len(nums)
        if nums[1]==nums[0]:
            dp[1]=True
        if nums[0]==nums[1]==nums[2] or nums[0]+1==nums[1]==nums[2]-1:
            dp[2]=True
        for i in range(3,len(nums)):
            if dp[i-2] and nums[i]==nums[i-1]:
                dp[i]=True
            elif dp[i-3] and nums[i]==nums[i-1]==nums[i-2]:
                dp[i]=True
            elif dp[i-3] and nums[i]==nums[i-1]+1==nums[i-2]+2:
                dp[i]=True
        return dp[-1]
    else:
        return nums[0]==nums[1]


# def back(s):
#     ss=[]
#     ss+=s
#     for i in range(len(ss)):
#         if i==0 and ss[i]=='#':
#             ss[i]='*'
#         elif ss[i]=='#':
#             ss[i]='*'
#             start_zhen=i-1
#             while start_zhen>0 and ss[start_zhen]=='*':
#                 start_zhen-=1
#             ss[start_zhen]='*'
#     res=[]
#     for item in ss:
#         if item!='*':
#             res.append(item)
#     return res
# print(back("#a#c"))


def minRemoveToMakeValid(s):
    zuo=[]
    you=[]
    for i in range(len(s)):
        if s[i]=='(':
            zuo.append(i)
        elif s[i]==')':
            if zuo:
                zuo.pop()
            else:
                you.append(i)
    mid=set(list(set(zuo))+list(set(you)))
    s1=''
    for i in range(len(s)):
        if i not in mid:
            s1+=s[i]
    return s1

def clumsy(n):
    dp1=list(range(n,0,-1))
    dp2=['*','//','+','-']
    s=''
    start_zhen=0
    while start_zhen<len(dp1):
        s+=str(dp1[start_zhen])
        if start_zhen!=len(dp1)-1:
            s+=dp2[start_zhen%4]
        start_zhen+=1
    return eval(s)


# class FreqStack:
#     def __init__(self):
#         self.s=[]
#         self.dic_c=SortedDict()
#     def push(self, val: int) -> None:
#         self.s.append(val)
#         self.dic_c[val]=self.dic_c.get(val,0)+1
#         print(self.s)
#         print(self.dic_c)
#     def pop(self) -> int:
#         s1=list(self.dic_c)
#         start_zhen=len(self.s)-1
#         while start_zhen>=0:
#             if self.dic_c[self.s[start_zhen]]==self.dic_c[s1[0]]:
#                 res=self.s[start_zhen]
#                 self.s.pop(start_zhen)
#                 self.dic_c[res]-=1
#                 if self.dic_c[res]==0:
#                     del self.dic_c[res]
#                 return res
#             start_zhen-=1
def makeGood(s:str,k) -> str:
    c=2
    while c==2:
        c=0
        i=0
        while i<=len(s)-k:
            cc=s[i:i+k]
            if s[i]==s[i+1] and cc.count(cc[0])==k:
                c=2
                ss=s[i:i+k]
                s=s.replace(ss,'')
                break
            i+=1
    return s

def removeDuplicates(s: str, k: int) -> str:
    ss=''
    for item in s:
        if not ss:
            ss+=item
        else:
            if ss and item!=ss[-1]:
                ss+=item
    for i in ss:
        s = "".join(s.split(i * k))
    return s


def removeOuterParentheses(s):
    stack=[]
    res=[]
    for i in range(len(s)):
        if s[i]=='(':
            stack.append(i)
        else:
            if s[i]==')' and len(stack)>1:
                stack.pop()
            else:
                res.append(stack.pop())
                res.append(i)
    ss=''
    for i in range(len(s)):
        if i not in res:
            ss+=s[i]
    return ss



def makeGood1(s: str) -> str:
    ss = ''
    for item in s:
        if not ss:
            ss += item
        else:
            if ss and item != ss[-1]:
                ss += item
    for i in ss:
        s = "".join(s.split(i * 2))
    return s
print(makeGood1(""))
