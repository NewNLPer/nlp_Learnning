# -*- coding:utf-8 -*-
# @Time      :2022/4/29 10:02
# @Author    :Riemanner
# Write code with comments !!!
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
# print(findTargetSumWays([1,1,1,1,1],3))
import copy
# def partition(s):
#     res = []
#     def backtrack(s, startIndex, path):
#         if startIndex>=len(s) and len(path)==3 and int(path[0])+int(path[1])==int(path[2]):
#             if (len(path[0])>1 and path[0][0]=='0') or (len(path[1])>1 and path[1][0]=='0') or (len(path[2])>1 and path[2][0]=='0'):
#                 return
#             else:
#                 res.append([int(path[0]),int(path[1]),int(path[2])])
#         if len(path)>3:
#             return
#         for i in range(startIndex, len(s)):
#             if len(res)!=0:
#                 break
#             if s[i]!=0:
#                 p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#                 path.append(p)  # 是回文子串
#                 backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#                 path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(s, 0, [])
#     if len(res)==0:
#         return []
#     else:
#         return res[0]
# print(partition("11235813"))
# class Solution(object):
#     def combinationSum4(self, nums, target):
#         self.dp = [-1] * (target + 1)
#         self.dp[0] = 1
#         return self.dfs(nums, target)
#
#     def dfs(self, nums, target):
#         if target < 0: return 0
#         if self.dp[target] != -1:
#             return self.dp[target]
#         res = 0
#         for num in nums:
#             res += self.dfs(nums, target - num)
#         self.dp[target] = res
#         return res
# print(Solution().combinationSum4([1,1,1,1,1],3))
# def fun(nums,target):
#     nums.sort(reverse = True)
#     if len(nums)==1 and abs(nums[0])==abs(target):
#         return 1
#     elif len(nums)==1 and abs(nums[0])!=abs(target):
#         return 0
#     else:
#         n=len(nums)-1
#         pa=fun(nums[:n],target+nums[-1])+fun(nums[:n],target-nums[-1])
#         return pa
# print(fun([1,0,0,0,0,0,0,0,0],1))

# def fun(nums,target):
#     if len(nums)==0 and target==0:
#         return 1
#     elif len(nums)==0 and target!=0:
#         return 0
#     else:
#         n=len(nums)-1
#         return fun(nums[:n],target+nums[-1])+fun(nums[:n],target-nums[-1])

# class Solution(object):
#     def combinationSum4(self, nums, target):
#         self.dp = [-1] * (target + 1)
#         self.dp[0] = 1
#         return self.dfs(nums, target)
#
#     def dfs(self, nums, target):
#         if target < 0: return 0
#         if self.dp[target] != -1:
#             return self.dp[target]
#         res = 0
#         for num in nums:
#             res += self.dfs(nums, target - num)
#         self.dp[target] = res
#         return res
# print(Solution().combinationSum4([1,2,3],4))
# import collections
# from functools import lru_cache
# class Solution:
#     def combinationSum4(self, nums,target):
#         dic = collections.defaultdict(int)
#         @lru_cache()
#         def dfs(target):
#             res = 0
#             if target == 0:
#                 return 1
#             if target < 0:
#                 return 0
#             for i in range(len(nums)):
#                 target_ = target - nums[i]
#                 res += dfs(target_) if target_ not in dic else dic[target_]
#             dic[target] = res
#             return res
#         return dfs(target)

# print(Solution().combinationSum4([1,2,3],4))

######组合和
class Solution:
    def dfs(self,target, nums, mry):
        if target<0:
            return 0
        if target==0:
            return 1
        if target in mry.keys():
            return mry[target]
        ans = 0
        for i in range(len(nums)):
            ans += self.dfs(target - nums[i], nums, mry)
        mry[target] = ans
        print(mry)
        return ans
    def findTargetSumWays(self, nums,target):
        mry = {}
        return self.dfs(target, nums, mry)
print(Solution().findTargetSumWays([1,2,4],8))

# ####正负的组合和
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

#####阶乘
# class Solution:
#     def dfs(self,target,mry):
#         if target==1:
#             return 1
#         if target==2:
#             return 2
#         if target in mry.keys():
#             return mry[target]
#         ans=(self.dfs(target-1,mry)*target)
#         mry[target] = ans
#         print(mry)
#         return ans
#     def findTargetSumWays(self,target):
#         mry = {}
#         return self.dfs(target,mry)
# print(Solution().findTargetSumWays(10))

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

# ####正负的组合和
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

# class Solution:
#     def dfs(self,target,mry):
#         if target==1:
#             return 1
#         if target==2:
#             return 2
#         if target in mry.keys():
#             return mry[target]
#         ans=(self.dfs(target-1,mry)*target)
#         mry[target] = ans
#         print(mry)
#         return ans
#     def findTargetSumWays(self,target):
#         mry = {}
#         return self.dfs(target,mry)
# print(Solution().findTargetSumWays(10))