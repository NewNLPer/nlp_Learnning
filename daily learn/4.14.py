####今天进行回溯法的学习，争取学会举一反三
import copy
######关于n数之和的回溯
# def combinationSum(candidates, target):
#     def bt(candidates, path, target):
#         if sum(path) == target:  # path和等于target时，加入结果集
#             res.append(copy.deepcopy(path))
#             return
#         elif sum(path) > target:  # path和大于target时，剪枝
#             return
#         for i in range(0, len(candidates)):  # 遍历选择列表
#             path.append(candidates[i])  # 做出选择
#             bt(candidates[i:], path, target)  # 进入下一层，注意candidates[i:]保证同一层不同广度的选择列表里没有已经选择过的数
#             path.pop()  # 撤销选择
#     res = []  # 结果集
#     bt(candidates, [], target)  # 开始递归
#     return res  # 返回结果集
# print(combinationSum([2,3,5],8))
#####关于子集的回溯def combinationSum(candidates, target):
# def combinationSum(candidates):
#     def bt(candidates, path, start):
#         res.append(copy.deepcopy(path))
#         for i in range(start, len(candidates)):  # 遍历选择列表
#             path.append(candidates[i])  # 做出选择
#             bt(candidates, path, i+1)  # 进入下一层，注意candidates[i:]保证同一层不同广度的选择列表里没有已经选择过的数
#             path.pop()  # 撤销选择
#     res = []  # 结果集
#     bt(candidates, [], 0)  # 开始递归
#     return res  # 返回结果集
# print(combinationSum([1,2,3]))
#####关于排列组合
# def combinationSum(candidates, target):
#     def bt(candidates,path,target,start):
#         if len(path)==target:  # path和等于target时，加入结果集
#             if path not in res:
#                 res.append(copy.deepcopy(path))
#             return
#         for i in range(0, len(candidates)):  # 遍历选择列表
#             path.append(candidates[i])  # 做出选择
#             bt(candidates[0:i]+candidates[i+1:], path, target,i+1)  # 进入下一层，注意candidates[i:]保证同一层不同广度的选择列表里没有已经选择过的数
#             path.pop()  # 撤销选择
#     res = []  # 结果集
#     bt(candidates,[], target,0)  # 开始递归
#     return res  # 返回结果集
# print(combinationSum([1,1,2],3))
# def combinationSum2(candidates,target):
#     if len(candidates) == 1:
#         if candidates[0] == target:
#             return [target]
#         else:
#             return []
#     else:
#         res = []
#         candidates.sort()
#         def bt(candidates, path, target,s):
#             if sum(path) == target and path not in res:
#                 res.append(copy.deepcopy(path))
#                 return
#             elif sum(path) > target:
#                 return
#             for i in range(s,len(candidates)):
#                 path.append(candidates[i])
#                 bt(candidates, path, target,i+1)
#                 path.pop()
#         bt(candidates, [], target,0)
#         return res
# print(combinationSum2([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],28))
# def combinationSum2(candidates,target):
#     if len(candidates) == 1:
#         if candidates[0] == target:
#             return [[target]]
#         else:
#             return []
#     elif sum(candidates) < target:
#         return []
#     elif sum(candidates) == target:
#         return [candidates]
#     elif sum(candidates)>target and len(set(candidates))==1:
#         if target%candidates[0]!=0:
#             return []
#         else:
#             return [[candidates[0]]*(target//candidates[0])]
#     else:
#         res = []
#         candidates.sort()
#         def bt(candidates, path, target, s):
#             if sum(path) == target and path not in res:
#                 res.append(copy.deepcopy(path))
#                 return
#             elif sum(path) > target:
#                 return
#             for i in range(s, len(candidates)):
#                 path.append(candidates[i])
#                 bt(candidates, path, target, i + 1)
#                 path.pop()
#
#         bt(candidates, [], target, 0)
#         return res
# print(combinationSum2([1,1,2,5,6,7,10],8))

# def combine(n,k):
#     if k == 1:
#         s = []
#         for i in range(1, n + 1):
#             s.append([i])
#         return s
#     elif k > n:
#         return []
#     elif k == n:
#         s = []
#         for i in range(1, n + 1):
#             s.append(i)
#         return [s]
#     else:
#         def bt(n, path, k, start):
#             if len(path) == k:
#                 res.append(copy.deepcopy(path))
#                 return
#             for i in range(start, n+1):
#                 path.append(i)
#                 bt(n, path, k, i + 1)
#                 path.pop()
#         res=[]
#         bt(n, [], k, 1)
# #         return res
# def zuhe(num):
#     res=[]
#     n=len(num)
#     def bt(num,path):
#         if len(path)==n:
#             res.append(copy.deepcopy(path))
#         for i in range(len(num)):
#             path.append(num[i])
#             bt(num[0:i]+num[i+1:],path)
#             path.pop()
#     bt(num,[])
#     return res
# print(zuhe([1,2,3]))
# def combinationSum(candidates):
#     def bt(candidates,path,start):
#         if len(path)==len(candidates):  # path和等于target时，加入结果集
#             if path not in res:
#                 res.append(copy.deepcopy(path))
#             return
#         for i in range(0, len(candidates)):  # 遍历选择列表
#             path.append(candidates[i])  # 做出选择
#             bt(candidates[0:i]+candidates[i+1:], path,i+1)  # 进入下一层，注意candidates[i:]保证同一层不同广度的选择列表里没有已经选择过的数
#             path.pop()  # 撤销选择
#     res = []  # 结果集
#     bt(candidates,[],0)  # 开始递归
#     return res  # 返回结果集
# print(combinationSum([1,2,3]))

# def xiangtong(s1,s2):
#     if len(s1)!=len(s2):
#         return False
#     else:
#         c1={}
#         for i in s1:
#             if i not in c1:
#                 c1[i] = 1
#             elif i in c1:
#                 c1[i]=c1[i]+1
#         c2={}
#         for i in s2:
#             if i not in c2:
#                 c2[i] = 1
#             elif i in c2:
#                 c2[i]=c2[i]+1
#     if c1==c2:
#         return True
#     else:
#         return False
# def groupAnagrams(strs):
#     sss = []
#     if len(strs)==0:
#         return sss
#     elif len(strs)==1:
#         return [strs]
#     else:
#         ss = []
#         ss.append(strs[0])
#         for j in strs[1:]:
#             if xiangtong(j,strs[0]) == True:
#                 ss.append(j)
#         for i in ss:
#             strs.remove(i)
#         sss.append(ss)
#         sss = sss + groupAnagrams(strs)
#     return sss
#
#
# strs=['da','ad','dada','dad']
#
# c=''.join(sorted('asdcvf'))
# print(c)
# def search(nums,target) -> int:
#     if len(nums) == 1:
#         if target == nums[0]:
#             return 0
#         else:
#             return -1
#     else:
#         i = 0  ##左指针
#         j = len(nums) - 1  # 右指针
#         while i <=j:
#             mid = (i + j) // 2
#             if nums[mid] == target:
#                 return mid
#             if nums[mid] >= nums[i]:  ####左侧有序
#                 if target >=nums[i] and target <nums[mid]:
#                     j = mid-1
#                 else:
#                     i = mid+1
#             elif nums[mid]<=nums[j]:
#                 if target <= nums[j] and  target >nums[mid]:
#                     i = mid+1
#                 else:
#                     j = mid-1
#         return -1
# print(search([4,5,6,7,0,1,2],8))






##########################################################################################################
####进行二分查找的学习，还有回溯算法的学习，首先对于递归仅仅说是一种简单的回溯，而回溯才是重点，对于二分查找还要注意，多练习，双指针
####首先对于二分查找的练习要注意，下面要进行关于回溯算法的一些练习，再加强一下
############子集问题，先画出图，下一层（i+1）+++元素减少（(start，len(nums))）
# def ziji(num):
#     res=[]
#     def bt(num,path,start):
#         if path not in res:
#             res.append(copy.deepcopy(path))
#         for i in range(start,len(num)):
#             path.append(num[i])
#             bt(num,path,i+1)
#             path.pop()
#
#     bt(num,[],0)
#     return res
# print(ziji([1,2,2]))
########组合问题
# def zuhe(n,k):
#     res=[]
#     def bt(n,k,path,start):
#         if len(path)==k:
#           res.append(copy.deepcopy(path))
#           return
#         for i in range(start,n+1):
#             path.append(i)
#             bt(n,k,path,i+1)
#             path.pop()
#     bt(n,k,[],1)
#     return res
# print(zuhe(4,2))
# def combinationSum4(nums, target):
#     if len(nums) == 1 and nums[0] > target:
#         return 0
#     elif len(nums) == 1 and nums[0]==target:
#         return 1
#     else:
#         res = []
#         def bt(nums, path, target):
#             if sum(path) == target:
#                 res.append(copy.deepcopy(path))
#                 return
#             elif sum(path) > target:
#                 return
#             for i in range(len(nums)):
#                 path.append(nums[i])
#                 bt(nums, path, target)
#                 path.pop()
#         bt(nums, [], target)
#         return len(res)
# print(combinationSum4([1,2,4],8))
# def firstMissingPositive(nums) -> int:
#     if len(nums) == 1:
#         if nums[0] < 0 or nums[0] == 0:
#             return 1
#         elif nums[0] > 1:
#             return 1
#     else:
#         for i in nums:
#             if i <= 0 or i > len(nums):
#                 continue
#             else:
#                 nums[i] = i
#     for i in nums:
#         if nums[i]!=i:
#             return i
#         else:
#             return len(nums)
# print(firstMissingPositive([2,3,4]))

def firstMissingPositive(nums):
    if len(nums) == 1:
        if nums[0] < 0 or nums[0] == 0:
            return 1
        elif nums[0] > 1:
            return 1
    else:
        for i in nums:
            if i <= 0 or i>=len(nums):
                continue
            else:
                nums[i] = i
        for i in range(1,len(nums)):
            if nums[i]!=i:
                return i
        else:
            return len(nums)
print(firstMissingPositive([1,2,0]))