# -*- coding:utf-8 -*-
# @Time      :2022/5/5 10:49
# @Author    :Riemanner
# Write code with comments !!!
#####关于两个子集的问题，需要再次进行重复
# def huocha(nums):
#     if sum(nums)%4!=0:
#         return False
#     c=int(sum(nums)/4)
#     if max(nums)>c:
#         return False
#     nums.sort()
#     k=4
#     while nums[-1]==c:
#         nums.pop()
#         k-=1
#     nums.sort(reverse=True)
#     s=[0]*k
#     def bt(inx,nums):
#         if inx==len(nums):
#             return True
#         for i in range(k):
#             if s[i]+nums[inx]<=c:
#                 s[i]+=nums[inx]
#                 if bt(inx+1,nums):
#                     return True
#                 s[i]-=nums[inx]
#         return False
#
#     return bt(0,nums)
# print(huocha([3,3,3,3,4]))
# import math
# ###有重复元素的排列组合
# def numSquarefulPerms(nums):
#     res = []
#     def bt(nums, path):
#         if not nums and path not in res:
#             res.append(path[:])
#         for i in range(len(nums)):
#             if len(path) == 0:
#                 path.append(nums[i])
#                 bt(nums[:i] + nums[i + 1:], path)
#                 path.pop()
#             elif len(path) >= 1 and math.sqrt(nums[i] + path[-1]) * math.sqrt(nums[i] + path[-1]) == nums[i] + path[-1]:
#                 path.append(nums[i])
#                 bt(nums[:i] + nums[i + 1:], path)
#                 path.pop()
#     bt(nums, [])
#     return res
# print(numSquarefulPerms([2,2,2,2,2,23,33,1,32,1,312,1,23,4]))
# import math
# def permuteUnique(nums):
#     nums.sort()
#     if nums[0]==nums[-1]:
#         if pow(int(math.sqrt(nums[0]+nums[1])),2)==nums[0]+nums[1]:
#             return 1
#         else:
#             return 0
#     elif len(set(nums))==2:
#         if pow(int(math.sqrt(nums[0]+nums[-1])),2)==nums[0]+nums[-1] and nums[0]!=:
#     else:
#         def bt(nums, path):
#             if not nums and path not in res:  # 递归到最底层，path加入结果集
#                 res.append(path[:])
#                 return
#             for i in range(len(nums)):
#                 if len(path)==0 or (len(path)>=1 and pow(int(math.sqrt(nums[i]+path[-1])),2)==nums[i]+path[-1]):
#                     path.append(nums[i])  # 做出选择
#                     bt(nums[:i] + nums[i + 1:], path)  # 进入下一层，选择列表不包含当前元素
#                     path.pop()  # 撤销选择
#         res = []  # 结果集
#         bt(nums, [])  # 开始递归
#         return res  # 返回结果集
# print(permuteUnique([0,0,0,0,0,0,1,1,1,1,1,1]))
import math
class Solution:
    def numSquarefulPerms(self, nums):
        res = 0
        n = len(nums)
        nums.sort()
        def isSqr(n):
            a = int((math.sqrt(n)))
            return a * a == n
        def dfs(visit,cur):
            nonlocal res
            if len(cur) == len(nums):
                res += 1
                return
            for i in range(len(nums)):
                if visit[i]: continue######该位置已经动过，选择跳过
                if i > 0 and nums[i] == nums[i-1] and visit[i-1]: #重复的数数已经在该位置放过，选择下一个
                    continue
                if len(cur) > 0 and isSqr(cur[-1] + nums[i]) == False:#当前的数并不满足关系，选择下一个
                    continue
                ####找到满足已条件的数字
                visit[i] = True ###记录已经进去
                cur.append(nums[i])###选择添加
                dfs(visit,cur)#####撤回操作
                visit[i] = False
                cur.pop()
        visit = [False] * n
        dfs(visit,[])
        return res
print(Solution().numSquarefulPerms([2,2,2]))
# def fun(nums):
#     S=[]
#     S+=nums
#     res=[]
#     s={}
#     def bt(path,S):
#         if (not S or len(path)>0) and path not in res:
#             res.append(path[:])
#         for i in range(len(S)):
#             if i==0 or S[i]!=S[i-1]:
#                 path.append(S[i])
#                 bt(path,S[:i]+S[i+1:])
#                 path.pop()
#     bt([],S)
#     return len(res)
# print(fun('CDC'))
# s='adas'
# c=[]
# c+=s
# def tribonacci(n):
#     s = {0:0,1:1,2:1}
#     def bt(n):
#         if n in s:
#             return s[n]
#         s[n] = bt(n - 1) + bt(n - 2) + bt(n - 3)
#         print(s)
#         return s[n]
#     print(s)
#     return bt(n)
# print(tribonacci(40))
# def maxSubArray(nums):
#     #####前n和＞0可以加加试试能不能更大，反正都一直记录，但是小于零，就立刻停止
#     result = nums[0]
#     tmp = nums[0]
#     for i in range(1, len(nums)):
#         if tmp > 0:
#             tmp = tmp + nums[i]
#         else:
#             tmp = nums[i]
#         result = max(tmp, result)
#     return result
# print(maxSubArray(nums = [-2,1,-3,4,-1,2,1,-5,4]))
s=[1,2,3]
b=dict(zip(s,[1,1,1]))
print(b)

