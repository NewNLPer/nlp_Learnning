# -*- coding:utf-8 -*-
# @Time      :2022/5/2 9:37
# @Author    :Riemanner
# Write code with comments !!!

#     def wordBreak(self,s,wordDict) -> bool:
#         memo = [None] * len(s)
#         # 以索引i为起始到末尾的字符串能否由字典组成
#         def dfs(i):
#             # 长度超过s,返回True(空字符能组成)
#             if i >= len(s):
#                 return True
#             # 存在以i为起始的递归结果
#             if memo[i] != None:
#                 return memo[i]
#             # 递归
#             for j in range(i, len(s)):
#                 ss=s[i:j+1]
#                 if ss in wordDict and dfs(j + 1):
#                     memo[i] = True
#                     return True
#             memo[i] = False
#             return False
#         return dfs(0)
# print(Solution().wordBreak("aab",["a","aa","aaa","aaaa"]))
####斐波那契数列
# class Solution:
#     def dfs(self,target,mry):
#         if target==1:
#             return 1
#         if target==2:
#             return 1
#         if target in mry.keys():
#             return mry[target]
#         ans=self.dfs(target-1,mry)+self.dfs(target-2,mry)
#         mry[target] = ans
#         print(mry)
#         return ans
#     def findTargetSumWays(self,target):
#         mry = {}
#         return self.dfs(target,mry)
# print(Solution().findTargetSumWays(10))
# def zuhe(s,w):
#     if len(s)==0 or s in w:
#         return True
#     for i in range(0,len(s)+1):
#         if s[0:i] in w :
#             return zuhe(s[i:],w)
#     return False
#
# print(zuhe('aaaaaaa',['aaaa','aaa']))

# class Solution:
#     def wordBreak(self, s: str, wordDict) -> bool:
#         memo = [None] * len(s)
#         # 以索引i为起始到末尾的字符串能否由字典组成
#         def dfs(i):
#             # 长度超过s,返回True(空字符能组成)
#             if i >= len(s):
#                 return True
#             # 存在以i为起始的递归结果
#             if memo[i] != None:
#                 return memo[i]
#             # 递归
#             for j in range(i, len(s)):
#                 if s[i:j + 1] in wordDict and dfs(j + 1):
#                     memo[i] = True
#                     print(memo)
#                     return True
#             memo[i] = False
#             return False
#         return dfs(0)
# print(Solution().wordBreak('aab',['a','aa']))
# def zuhe(s,w):
#     c=[None]*len(s)
#     def dfs(i):
#         if i >=len(s):
#             return True
#         if c[i]!=None:
#             return c[i]
#         for j in range(i,len(s)):
#             if s[i:j+1] in w and dfs(j+1):
#                 c[i]=True
#                 return True
#         c[i]=False
#         return False
#     return dfs(0)
# print(zuhe(s = "aaaaaaa", w = ['aaa','aaaa']))
import copy
# def wordBreak(s,wordDict,):
#     res = []
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):   # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res.append(' '.join(path))
#         for i in range(startIndex, len(s)):
#             c=s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             if c in wordDict :
#                 path.append(c)  # 是回文子串
#             else:
#                 continue  # 不是回文，跳过
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(s, 0, [])
#     return res
# print(wordBreak(s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]))

# def zuhe(s):
#     def dfs(i):
#         if i >=len(s):
#             return True
#         for j in range(i,len(s)):
#             s1=s[j]
#             s2=
#             if s[i:j+1] in w and dfs(j+1):
#                 c[i]=True
#                 return True
#         c[i]=False
#         return False
#     return dfs(0)
# print(zuhe(s = "aaaaaaa", w = ['aaa','aaaa']))
# def wordBreak(s,wordDict,):
#     res = [0]
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):   # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res[0]=1
#         for i in range(startIndex, len(s)):
#             c=s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             if c in wordDict :
#                 path.append(c)  # 是回文子串
#             else:
#                 continue  # 不是回文，跳过
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(s, 0, [])
#     return res
# print(wordBreak(s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]))
# def partition( num: str):
#     res = []
#     def backtrack(num, startIndex, path):
#         if startIndex >= len(num) and len(path)>2:  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res.append(path[:])
#         for i in range(startIndex, len(num)):
#             p = num[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             if len(path)>=2 and int(path[-1])+int(path[-2])==int(p) and str(int(p))==p and int(p)<pow(2,31):
#                 path.append(int(p))  # 是回文子串
#             elif len(path)<2 and str(int(p))==p and int(p)<pow(2,31):
#                 path.append(int(p))
#             else:
#                 continue
#             backtrack(num, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(num, 0, [])
#     if len(res)!=0:
#         return res[0]
#     else:
#         return []
# print(partition("539834657215398346785398346991079669377161950407626991734534318677529701785098211336528511"))

#####链表反转，关于链表的处理题目都是在这个的基础上进行的
# class Node():
#     def __init__(self,val):
#         self.val=val
#         self.next=None
# a=Node(1)
# b=Node(2)
# c=Node(3)
# d=Node(4)
# a.next=b
# b.next=c
# c.next=d
# def dayin(head):
#     while head:
#         print(head.val,end=' ')
#         head=head.next
# dayin(a)
#
# def fanzhu(head):
#     pre=None
#     cur=head
#     while cur:
#         tmp=cur.next
#         cur.next=pre
#         pre=cur
#         cur=tmp
#     return pre
# print()
# cc=fanzhu(a)
# dayin(cc)
# def partition(s):
#     res = [0]
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res[0]+=1
#         for i in range(startIndex, len(s)):
#             p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             if int(p)<=26 and int(p)!=0 and str(int(p))==p:
#                 path.append(p)
#             else:
#                 continue  # 不是回文，跳过
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(s, 0, [])
#     return res[0]
# print(partition('111121111111111251411314141231111111111'))




# def sm(s):
#     if s[0]=='0':
#         return 0
#     else:
#         first_tmp=0
#         second_tmp=1
#         ans=0
#         while second_tmp<len(s):
#             tmp=s[first_tmp]+s[second_tmp]
#             if int(tmp)<=26:
#                 if tmp=='10' or tmp=='20':
#                     ans=ans+1
#                     first_tmp=second_tmp
#                     second_tmp+=1
#                 elif str(int(tmp))==tmp:
#                     ans=ans+2
#                     first_tmp=second_tmp
#                     second_tmp+=1
#                 elif str(int(tmp))!=tmp:
#                     ans=ans+0
#                     first_tmp=second_tmp
#                     second_tmp+=1
#             else:
#                 ans=ans+1
#                 first_tmp=second_tmp
#                 second_tmp+=1
#         return ans-(len(s)-2)
# print(sm("1"))
# class Node():
#     def __init__(self,val):
#         self.val=val
#         self.next=None
# a=Node(4)
# b=Node(1)
# c=Node(3)
# d=Node(2)
# a.next=b
# b.next=c
# c.next=d
# def dayin(head):
#     while head:
#         print(head.val,end=' ')
#         head=head.next
# dayin(a)






##处理列表
def fun(nums):
    s=[1]
    start_zhi=0
    end_zhi=1
    while end_zhi<len(nums):
        s.append(nums[start_zhi]+nums[end_zhi])
        start_zhi=end_zhi
        end_zhi+=1
    s.append(1)
    return s
# from functools import lru_cache
# @lru_cache()
def generate( numRows):
    if numRows == 1:
        return [[1]]
    if numRows == 2:
        return [[1], [1, 1]]
    else:
        return generate(numRows - 1) + [fun(generate(numRows - 1)[-1])]
print(generate(30))




