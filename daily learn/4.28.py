# -*- coding:utf-8 -*-
# @Time      :2022/4/28 10:53
# @Author    :Riemanner
# Write code with comments !!!
###回溯法解决电话号码组合问题
import copy

####老回溯
# def letterCombinations(nums,target):
#     def bt(nums, path, start,target):###强调剪枝的重要性,不要乱返回
#         if sum(path)==target and len(path)==len(nums):
#             res[0]+=1
#             return
#         for i in range(start,len(nums)):
#             bian=[-nums[i],nums[i]]
#             for j in range(2):
#                 path.append(bian[j])
#                 bt(nums,path,i+1,target)
#                 path.pop()
#     res=[0]
#     bt(nums,[],0,target)
#     return res[0]
# print(letterCombinations([1,1,1,1,1],3))
# ###新回溯
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
# print(xinhuisu([1,1,1,1,1],3))
# import copy
# def letterCombinations(s):
#     def bt(s, path, start):###强调剪枝的重要性,不要乱返回
#         if len(path)==len(s):
#             res.append(copy.deepcopy(path))
#             return
#         for i in range(start,len(s)):
#             if s[i].isdigit():
#                 path=path+s[i]
#                 bt(s,path,i+1)
#                 path=path[:len(path)-1]
#             else:
#                 bian=list({s[i],s[i].upper(),s[i].lower()})
#                 for j in range(2):
#                     path=path+bian[j]
#                     bt(s,path,i+1)
#                     path=path[:len(path)-1]
#     res=[]
#     bt(s,'',0)
#     return res
# print(letterCombinations('C'))
# def uqnapailei(nums):
#     res=[]
#     def bt(nums,path):
#         if len(path)==len(nums):
#             res.append(path[:])
#         for i in range(len(nums)):
#             if nums[i] not in path:
#                 path.append(nums[i])
#                 bt(nums,path)
#                 path.pop()
#     bt(nums,[])
#     return res
# print(uqnapailei([1,2,3]))
# def partition(s):
#     res = []
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res.append(path[:])
#         for i in range(startIndex, len(s)):
#             p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             print('=========')
#             print(p)
#             print('=========')
#             if p == p[::-1]:
#                 path.append(p)  # 是回文子串
#             else:
#                 continue  # 不是回文，跳过
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(s, 0, [])
#     return res
# print(partition('aabcd'))
def zuhe(s,wordDict):
    res=[]
    def bt(s,path,wordDict):
        if ''.join(path)==s:
            res.append(path[:])
            return True
        if len(''.join(path))>len(s):
            return
        for i in range(len(wordDict)):
            path.append(wordDict[i])
            bt(s,path,wordDict)
            path.pop()
    bt(s,[],wordDict)
    if len(res)==0:
        return False
    else:
        return True
print(zuhe("catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]))

