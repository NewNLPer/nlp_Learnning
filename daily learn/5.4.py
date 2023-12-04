# -*- coding:utf-8 -*-
# @Time      :2022/5/4 12:15
# @Author    :Riemanner
# Write code with comments !!!
###只考虑以下两点，1、集合+组合+排列 2、字符串切割回文
###排列问题
# def pailie(nums):
#     res=[]
#     def bt(path):
#         if len(path)==len(nums):
#             res.append(path[:])
#         for i in range(len(nums)):
#             if nums[i] not in path:
#                 path.append(nums[i])
#                 bt(path)
#                 path.pop()
#     bt([])
#     return res
# print(pailie([1,2,3]))
###集合问题
# def jihe(nums):
#     res=[]
#     def bt(start,path):
#         res.append(path[:])
#         for i in range(start,len(nums)):
#             path.append(nums[i])
#             bt(i+1,path)
#             path.pop()
#     bt(0,[])
#     return res
# def zongshu(nums,k):
#     res=[]
#     nums1=jihe(nums)
#     def bt(path,k,start):
#         if len(path)==k:
#             ss=[]
#             for i in path:
#                 ss=ss+i
#             ss.sort()
#             if ss==nums:
#                 res.append(path[:])
#             else:
#                 return
#         for j in range(start,len(nums1)):
#             path.append(nums1[j])
#             bt(path,k,j+1)
#             path.pop()
#     bt([],k,0)
#     return res
# print(zongshu([5,5,5,5,4,4,4,4,3,3,3,3],4))

###组合问题
# def zuhe(n,k):
#     res=[]
#     def bt(n,k,path,start):
#         if len(path)==k:
#             res.append(path[:])
#         if len(path)>k:
#             return
#         for i in range(start,n+1):
#             path.append(i)
#             bt(n,k,path,i+1)
#             path.pop()
#     bt(n,k,[],1)
#     return res
# print(zuhe(4,2))
# def zifuchuanqiege(s):
#     res=[]
#     def bt(start,path):
#         if start>=len(s) and len(path)==4 and sum(path[0])==sum(path[1])==sum(path[2])==sum(path[3]):
#             res.append(path[:])
#         for i in range(start,len(s)):
#             p=s[start:i+1]
#             path.append(p)
#             bt(i+1,path)
#             path.pop()
#     bt(0,[])
#     return res
# print(zifuchuanqiege([5,5,5,5,4,4,4,4,3,3,3,3]))

###组合问题
# def zuhe(nums,k):
#     res=[]
#     def bt(path,k,start):
#         if sum(path)==k:
#             res.append(path[:])
#         if sum(path)>k:
#             return
#         for i in range(start,len(nums)):
#             path.append(nums[i])
#             bt(path,k,i+1)
#             path.pop()
#     bt([],k,0)
#     return res
# print(zuhe([5,5,5,5,4,4,4,4,3,3,3,3],12))
# class Solution(object):
#     def makesquare(self, nums):
#         if len(nums) < 4:
#             return False
#         if sum(nums) % 4 != 0:
#             return False
#         l = sum(nums) // 4
#         mark = [False for _ in nums]
#         nums.sort()
#         nums.reverse()
#         def backTrack(ls, summary, mk):
#             if summary == l:
#                 ls += 1
#                 summary = 0
#             if ls == 4:
#                 return True
#             for i in range(len(nums)):
#                 if mk[i] == False and summary + nums[i] <= l:
#                     mk[i] = True
#                     if backTrack(ls, summary + nums[i], mk):
#                         return True
#                     if summary == 0:
#                         return False
#                     if summary + nums[i] == l:
#                         return False
#                     mk[i] = False
#         return backTrack(0, 0, mark)
# print(Solution().makesquare([1,1,2,2,2]))
# class Solution:
#     def makesquare(self, matchsticks):
#         board = int(sum(matchsticks) / 4)
#         matchsticks.sort(reverse=True)
#         if sum(matchsticks) % 4 != 0:
#             return False
#         elif matchsticks[0]>board:
#             return False
#         boards = [0, 0, 0, 0]
#         return self.dfs(matchsticks, boards, board, 0)
#     def dfs(self, matchsticks, boards, board, ind):
#         if ind == len(matchsticks):
#             return True
#         for i in range(4):
#             if boards[i] + matchsticks[ind] <= board:
#                 boards[i] += matchsticks[ind]
#                 if self.dfs(matchsticks, boards, board, ind + 1):
#                     return True
#                 boards[i] -= matchsticks[ind]
#         return False
# print(Solution().makesquare([1,1,2,2,2]))
# class Solution:
#     def canPartitionKSubsets(self, nums,k):
#         used = [False] * len(nums)
#         s = sum(nums)
#         if s % k != 0:
#             return False
#         target = s // k
#         nums.sort()
#         while nums and nums[-1] >= target:
#             if nums[-1] > target:
#                 return False
#             nums.pop()
#             k -= 1
#         nums.reverse()
#         return self.backtrack(nums, used, k, 0, 0, target)
#     def backtrack(self, nums, used, k, bucket, start, target):
#         if k == 0:
#             return True
#         if bucket == target:
#             return self.backtrack(nums, used, k - 1, 0, 0, target)
#         for i in range(start, len(nums)):
#             if used[i]:
#                 continue
#             if bucket + nums[i] > target:
#                 continue
#             used[i] = True
#             bucket += nums[i]
#             if self.backtrack(nums, used, k, bucket, i + 1, target):
#                 return True
#             used[i] = False
#             bucket -= nums[i]
#         return False
# def canPartitionKSubsets(nums,k):
#     n = len(nums)
#     tot_sum = sum(nums)
#     if k > n or tot_sum % k != 0:
#         return False
#     target = tot_sum / k  # 每个子集的和
#     nums.sort(reverse=True)
#     # sums = [0 for _ in range(k)]
#     #####小剪枝，先把等于target 的先挑选出来
#     while nums and nums[-1]==target:
#         nums.pop()
#         k-=1

#
# def canPartitionKSubsets(nums,k):
#     if sum(nums)%k!=0:
#         return False
#     nums.sort()
#     target=int(sum(nums)/k)
#     if nums[0]>target:
#         return False
#     while nums and nums[-1]==target:
#         nums.pop()
#         k-=1
#     nums.sort(reverse=True)
#     s=[]
#     for i in range(k):
#         s.append(0)
#     def bt(inx,board,k,sss):
#         if inx==len(nums):
#             return True
#         for i in range(k):
#             if board[i]+nums[inx]<=target :
#                 board[i]+=nums[inx]
#                 if bt(inx+1,board,k,sss):
#                     return sss[inx]
#                 board[i] -= nums[inx]
#         return False
#     return bt(0,s,k,{})
# print(canPartitionKSubsets([3,9,4,5,8,8,7,9,3,6,2,10,10,4,10,2],10))
# def zuhecha(n,k):
#     res=[]
#     def bt(path):
#         if len(path)==n:
#             res.append(path[:])
#         if len(path)>n:
#             return
#         for i in range(0,10):
#             if len(path)>=1 and abs(i-int(path[-1]))==k and str(int(path+str(i)))==path+str(i):
#                 path=path+str(i)
#                 bt(path)
#                 path=path[:len(path)-1]
#             elif len(path)==0:
#                 path=path+str(i)
#                 bt(path)
#                 path=path[:len(path)-1]
#             else:
#                 continue
#     bt('')
#     return list(map(lambda x: int(x), res))
#
# print(zuhecha(9,9))
# def zifuchuan(s):
#     res=[]
#     def bt(path,start):
#         if start>=len(s) and len(path)==2:
#             res.append(path[:])
#         for i in range(start,len(s)):
#             p=s[start:i+1]
#             if str(int(p))==p:
#                 path.append(p)
#                 bt(path,i+1)
#                 path.pop()
#     bt([],0)
#     return res
# print(zifuchuan('0123'))
# def paile(nums):
#     res=[]
#     def bt(start,path):
#         if len(path)<3 :
#             res.append(path[:])
#         for i in range(start,len(nums)):
#             path.append(nums[i])
#             bt(i+1,path)
#             path.pop()
#     bt(0,[])
#     return res
# def numTilePossibilities(tiles):
#     if len(tiles) == 1:
#         return [tiles]
#     elif len(tiles) == 2:
#         return [tiles[0], tiles[1], tiles, tiles[::-1]]
#     else:
#         S=[]
#
#         for i in range(len(tiles)-1,-1,-1):
#             c = tiles[i]
#             cc = tiles[:i]
#             for k in numTilePossibilities(cc):
#                 S.append(c)
#                 S.append(k)
#                 S.append(c + k)
#                 S.append(k + c)
#                 for j in range(len(i)):
#                     S.append(i[0:j] + c + i[j:])
#         return len(set(S))
# print(numTilePossibilities('aaasd'))
def pailie(nums):
    res=[]
    def bt(start,k,path):
        if len(path)==k:
            res.append(path[:])
        for i in range(start,len(nums)):
            path.append(nums[i])
            bt(i+1,k+1,path)
            path.pop()
    bt(0,0,[])
    return res
print(pailie([1,2,3]))