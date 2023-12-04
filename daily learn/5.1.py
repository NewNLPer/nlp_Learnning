# -*- coding:utf-8 -*-
# @Time      :2022/5/1 14:22
# @Author    :Riemanner
# Write code with comments !!!
###正负的和关于回溯的三种方法
#1、双线路回溯##############################
# def xinhuisu(nums,target):
#     res=[0]
#     def bt(nums,start,path,target):
#         if sum(path)==target and len(path)==len(nums):
#             res[0]+=1
#             return
#         for i in range(start,len(nums)):
#             bt(nums,i+1,path+[-nums[i]],target)
#             bt(nums,i+1,path+[nums[i]],target)
#     bt(nums,0,[],target)
#     return res[0]
# print(xinhuisu([1],1))
#2、纯回溯
# def findTargetSumWays(nums, target):
#     res = [0]
#     def bt(nums, path, start, target):  ###强调剪枝的重要性,不要乱返回
#         if sum(path) == target and len(path) == len(nums):
#             res[0] += 1
#             return
#         for i in range(start, len(nums)):
#             bian = [-nums[i], nums[i]]
#             for j in range(2):
#                 if sum(path)+bian[j]<=target:
#                     path.append(bian[j])
#                     bt(nums, path, i + 1, target)
#                     path.pop()
#                 else:
#                     continue
#
#     bt(nums, [], 0, target)
#     return res[0]
#3、带记忆的回溯
# class Solution:
#     def dfs(self, i, target, nums, mry):
#         # if i == len(nums):
#         #     if target == 0: return 1
#         #     else: return 0
#         if i == len(nums): return target == 0
#         if (i, target) in mry.keys(): return mry[(i, target)]
#         ans = 0
#         ans += self.dfs(i + 1, target - nums[i], nums, mry)
#         ans += self.dfs(i + 1, target + nums[i], nums, mry)
#         mry[(i, target)] = ans
#         return ans
#
#     def findTargetSumWays(self, nums: List[int], target: int) -> int:
#         mry = {}
#         return self.dfs(0, target, nums, mry)

######组合和
#1、带记忆的回溯
# class Solution:
#     def dfs(self,target, nums, mry):
#         if target<0:
#             return 0
#         if target==0:
#             return 1
#         if target in mry.keys():
#             return mry[target]
#         ans = 0
#         for i in range(len(nums)):
#             ans += self.dfs(target - nums[i], nums, mry)
#         mry[target] = ans
#         print(mry)
#         return ans
#     def findTargetSumWays(self, nums,target):
#         mry = {}
#         return self.dfs(target, nums, mry)
# print(Solution().findTargetSumWays([9],3))
##纯回溯
# def func(nums,target):
#     res=[]
#     def bt(nums,path,target):
#         if sum(path)==target:
#             res.append(path[:])
#         if sum(path)>target:
#             return
#         for i in range(len(nums)):
#             path.append(nums[i])
#             bt(nums,path,target)
#             path.pop()
#     bt(nums,[],target)
#     return res
# print(func([1,2,4],8))
####切割字符串的回文一直都非常重要
import copy
# class Solution:
#     def partition(self, s: str):
#         res = []
#         def backtrack(s,startIndex,path):
#             if startIndex >= len(s) and len(path)==4:  #如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#                 res.append(''.join(path))
#             for i in range(startIndex,len(s)):
#                 p = s[startIndex:i+1]  #获取[startIndex,i+1]在s中的子串
#                 if int(p)<=255 and str(int(p))==p:
#                     if i==len(s)-1:
#                         path.append(p)  #是回文子串
#                     else:
#                         path.append(p+'.')
#                 else:
#                     continue  #不是回文，跳过
#                 backtrack(s,i+1,path)  #寻找i+1为起始位置的子串
#                 path.pop()  #回溯过程，弹出本次已经填在path的子串
#         backtrack(s,0,[])
#         return res
# print(Solution().partition(s = "101023"))
# import copy
# class Solution:
#     def partition(self, s: str):
#         res = []
#         def backtrack(s,startIndex,path):
#             if startIndex >= len(s):  #如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#                 res.append(copy.deepcopy(path))
#             for i in range(startIndex,len(s)):
#                 p = s[startIndex:i+1]  #获取[startIndex,i+1]在s中的子串
#                 if p == p[::-1]:
#                     path.append(p)  #是回文子串
#                 else:
#                     continue  #不是回文，跳过
#                 backtrack(s,i+1,path)  #寻找i+1为起始位置的子串
#                 path.pop()  #回溯过程，弹出本次已经填在path的子串
#         backtrack(s,0,[])
#         return res
# class Solution:
#     def minCut(self, s: str) -> int:
#         res = [len(s)-1]
#         def backtrack(s, startIndex, path,cc):
#             if startIndex >= len(s) and len(path)-1<cc:  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#                 cc=min(cc,len(path)-1)
#                 res[0]=min(res[0],cc)
#             if len(path)-1>res[0]:
#                 return
#             for i in range(startIndex, len(s)):
#                 p=s[startIndex:i+1]
#                 if p == p[::-1]:
#                     path.append(p)  # 是回文子串
#                 else:
#                     continue  # 不是回文，跳过
#                 backtrack(s, i + 1, path,cc)  # 寻找i+1为起始位置的子串
#                 path.pop()  # 回溯过程，弹出本次已经填在path的子串
#         backtrack(s, 0, [],len(s)-1)
#         return res[0]
#
# class Solution:
#     def wordBreak(self, s, wordDict):
#         res=[]
#         def bt(s,path,wordDict):
#             if ''.join(path)==s:
#                 res.append(path[:])
#                 return
#             if len(''.join(path))>len(s):
#                 return
#             for i in range(len(wordDict)):
#                 if len(path)==0 and  s[len(path):len(path)+len(wordDict[i])]==wordDict[i]:
#                     path.append(wordDict[i])
#                     bt(s,path,wordDict)
#                     path.pop()
#                 elif s[len(path[-1]):len(path[-1])+len(wordDict[i])]==wordDict[i]:
#                     path.append(wordDict[i])
#                     bt(s, path, wordDict)
#                     path.pop()
#                 else:
#                     print(s[len(path[-1]):len(path[-1])+len(wordDict[i])])
#                     continue
#         bt(s,[],wordDict)
#         if len(res)==0:
#             return False
#         else:
#             return True
# print(Solution().wordBreak(s = "leetcode", wordDict = ["leet", "code"]))
# class Solution:
#     def wordBreak(self, s, wordDict):
#         res=[]
#         def bt(s,path,wordDict):
#             if ''.join(path)==s:
#                 res.append(path[:])
#                 return
#             if len(''.join(path))>len(s):
#                 return
#             for i in range(len(wordDict)):
#                 if wordDict[i] in s:
#                     path.append(wordDict[i])
#                     bt(s,path,wordDict)
#                     path.pop()
#                 else:
#                     continue
#         bt(s,[],wordDict)
#         if len(res)==0:
#             return False
#         else:
#             return True
# def zuhe(s,w):
#     def bt(s):
#         if len(s)==0:
#             return True
#         res=False
#         for i in range(1,len(s)+1):
#             if s[i] in w:
#                 res=bt(s[i+1:]) or res
#         return res
#     return bt(s)
# class Solution:
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
#                 if s[i:j + 1] in wordDict and dfs(j + 1):
#                     memo[i] = True
#                     return True
#             memo[i] = False
#             return False
#         return dfs(0)

def zuhe(s):
    def dfs(start,mome):
        if start>=len(s):
            return 1
        if s[start]=='0':
            return 0
        if start in mome:
            return mome[start]
        a=dfs(start+1,mome)
        b=0
        if int(s[start:start+2])<=26 and len(s)-start>=2:
            b=dfs(start+2,mome)
        mome[start]=a+b
        return a+b
    return dfs(0,{})
print(zuhe('11203'))
# def youmei(n):
#     res=[]
#     def bt(path,n):
#         if len(path)==n:
#             for i in range(len(path)):
#                 if path[i]%(i+1)==0 or (i+1)%path[i]==0:
#                     res.append(path[:])
#                 else:
#                     return
#         for i in range(1,n+1):
#             if i not in path:
#                 path.append(i)
#                 bt(path,n)
#                 path.pop()
#     bt([],n)
#     return len(res)
# print(youmei(3))


#
# def zuhe(s):
#     @lru_cache()
#     def dfs(start):
#         if start>=len(s):
#             return 1
#         if s[start]=='0':
#             return 0
#         a=dfs(start+1)
#         b=0
#         if int(s[start:start+2])<=26 and len(s)-start>=2:
#             b=dfs(start+2)
#         return a+b
#     return dfs(0)
# print(zuhe('111111111111111111111111111111111111111111'))
###for 加一个指针

