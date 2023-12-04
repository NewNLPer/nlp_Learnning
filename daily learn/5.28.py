# -*- coding:utf-8 -*-
# @Time      :2022/5/28 9:54
# @Author    :Riemanner
# # Write code with comments !!!
# def compareVersion(version1: str, version2: str) -> int:
#     if version1.count('.')==version2.count('.'):
#         while len(version1)!=0 and len(version2)!=0:
#             if version1.count('.')==version2.count('.')==0:
#                 if int(version1)==int(version2):
#                     return 0
#                 elif int(version1)<int(version2):
#                     return -1
#                 else:
#                     return 1
#             else:
#                 start_zhen1=0
#                 start_zhen2=0
#                 while start_zhen1<len(version1):
#                     if version1[start_zhen1]!='.':
#                         start_zhen1+=1
#                     else:
#                         break
#                 while start_zhen2<len(version2):
#                     if version2[start_zhen2]!='.':
#                         start_zhen2+=1
#                     else:
#                         break
#                 if int(version1[:start_zhen1])>int(version2[:start_zhen2]):
#                     return 1
#                 elif int(version1[:start_zhen1])<int(version2[:start_zhen2]):
#                     return -1
#                 else:
#                     version1=version1[start_zhen1+1:]
#                     version2=version2[start_zhen2+1:]
#     elif version1.count('.')>version2.count('.'):###补充再次调用
#         k=version1.count('.')-version2.count('.')
#         while k>0:
#             version2+='.0'
#             k-=1
#         return compareVersion(version1,version2)
#     elif version1.count('.')<version2.count('.'):###补充再次调用
#         k=version2.count('.')-version1.count('.')
#         while k>0:
#             version1+='.0'
#             k-=1
#         return compareVersion(version1,version2)
# print(compareVersion(version1 = "0.1", version2 = "1.1"))
# class Node():
#     def __init__(self,val):
#         self.val=val
#         self.next=None
# def dayin(head):
#     while head:
#         print(head.val,end=' ')
#         head=head.next
# s=[1,2,3,4]
# a=Node(s[0])
# tail=a
# for i in s[1:]:
#     tail.next=Node(i)
#     tail=tail.next
# dayin(a)
# def yici(head):
#     tail=head.next
#     tail1=tail
#     while tail1.next:
#         tail1=tail1.next
#     tail1.next=head
#     head.next=None
#     return tail
# cc=yici(a)
# print()
# dayin(cc)s
###元素字母相差一个，函数来定义



# def find_neighbours(a,b):  # 对于一个单个字符s, 从wordList找到所有相邻字符。
#     start_zhen1=0
#     flag=0
#     while start_zhen1<len(a):
#         if a[start_zhen1]==b[start_zhen1]:
#             flag+=1
#             start_zhen1+=1
#         else:
#             start_zhen1+=1
#     if flag==len(a)-1:
#         return True
#     else:
#         return False
# def ladderLength(beginWord,endWord,wordList):
#     if endWord not in wordList:
#         return 0
#     else:
#         res=[5001]
#         def bt(path,wordList):
#             if path and  path[-1]==endWord:
#                 res[0]=min(res[0],len(path))
#                 return
#             if not wordList or len(path)>res[0]:
#                 return
#             for i in range(len(wordList)):
#                 if find_neighbours(path[-1],wordList[i]):
#                     path.append(wordList[i])
#                     bt(path,wordList[0:i]+wordList[i+1:])
#                     path.pop()
#     bt([beginWord],wordList)
#     if res[0]==5001:
#         return 0
#     else:
# import math
# def countSubstrings(s: str) -> int:
#     res=[0]
#     def bt(path,start):
#         if start>=len(s):
#             res[0]+=len(path)
#             return
#         for i in range(start,len(s)):
#             p=s[start:i+1]
#             if p==p[::-1]:
#                 path.append(p)
#             else:
#                 continue
#             bt(path,i+1)
#             path.pop()
#     bt([],0)
#     return res[0]
# print(countSubstrings('aaa'))
# def countSubstrings(s: str) -> int:
#     res=len(s)
#     for i in range(2,len(s)+1):
#         start_zhen=0
#         end_zhen=start_zhen+i
#         while end_zhen<=len(s):
#             p=s[start_zhen:end_zhen]
#             if p==p[::-1]:
#                 res+=1
#                 start_zhen+=1
#                 end_zhen=start_zhen+i
#             else:
#                 start_zhen+=1
#                 end_zhen=start_zhen+i
#     return res
# def findDisappearedNumbers(nums):
#     nums1=[0]*len(nums)
#     for i in range(len(nums)):
#         nums1[nums[i]-1]=1
#     res=[]
#     for j in range(len(nums1)):
#         if nums1[j]==0:
#             res.append(j+1)
#     return res
# print(findDisappearedNumbers(nums = [4,3,2,7,8,2,3,1]))
# def fourSumCount(nums1,nums2,nums3,nums4):
#     s1={}
#     s2={}
#     res=0
#     for i in range(len(nums1)):
#         for j in range(len(nums2)):
#             s1[nums1[i]+nums2[j]]=s1.get(nums1[i]+nums2[j],0)+1
#     for i in range(len(nums3)):
#         for j in range(len(nums4)):
#             s2[nums3[i]+nums4[j]]=s2.get(nums3[i]+nums4[j],0)+1
#     for key in s1:
#         if s2.get(-key,0)!=0:
#             res+=s1[key]*s2[-key]
#     return res
# print(fourSumCount(nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]))
# def canPartition(nums) -> bool:
#     if sum(nums)%2!=0 or max(nums)>sum(nums)//2:
#         return False
#     elif len(nums)==2:
#         if nums[0]!=nums[1]:
#             return False
#         else:
#             return True
#     else:
#         board=sum(nums)//2
#         boards = [0]*2
#         def dfs(nums, boards, board, ind):
#             if ind == len(nums):
#                 return True
#             for i in range(len(boards)):
#                 if boards[i] + nums[ind] <= board:
#                     boards[i] += nums[ind]
#                     if dfs(nums, boards, board, ind + 1):
#                         return True
#                     boards[i] -= nums[ind]
#             return False
#         return dfs(nums, boards, board, 0)
# print(canPartition([3,3,3,4,5]))
# def isAnagram(s: str, t: str) -> bool:
#     s1={}
#     s2={}
#     for i in s:
#         s1[i]=s1.get(i,0)+1
#     for j in t:
#         s2[i]=s2.get(j,0)+1
#     for key in s1:
#         if s1[key]!=s2.get(key,0):
#             return False
# #     return True
# def findAnagrams(s: str, p: str):
#     dic_c1={}
#     dic_c2={}
#     res=[]
#     for i in p:
#         dic_c1[i]=dic_c1.get(i,0)+1
#     for j in s[:len(p)]:
#         dic_c2[j]=dic_c2.get(j,0)+1
#     if dic_c1==dic_c2:
#         res.append(0)
#     for k in range(1,len(s)-len(p)+1):
#         if dic_c2[s[k-1]]>1:
#             dic_c2[s[k-1]]-=1
#         else:
#             del dic_c2[s[k-1]]
#         dic_c2[s[k+len(p)-1]]=dic_c2.get(s[k+len(p)-1],0)+1
#         if dic_c1==dic_c2:
#             res.append(k)
#     return res
# print(findAnagrams(s = "abab", p = "ab"))

# def summaryRanges(nums):
#     res=[]
#     start_zhen=0
#     while start_zhen<len(nums):
#         p=str(nums[start_zhen])
#         end_zhen=start_zhen
#         while start_zhen<len(nums)-1 and nums[start_zhen+1]-1==nums[start_zhen]:
#             start_zhen+=1
#         if end_zhen==start_zhen:
#             res.append(p)
#             start_zhen+=1
#         else:
#             p+='->'+str(nums[start_zhen])
#             res.append(p)
#             start_zhen+=1
#     return res
# print(summaryRanges(nums = [0]))
# def checkSubarraySum(nums, k) -> bool:
#     sum, res, cul,s1 = 0, 0, {0:-1},{0}
#     for i in range(len(nums)):
#         sum += nums[i]
#         if sum%k in s1:
#             if i-cul[sum%k]<=1:
#                 continue
#             else:
#                 return True
#         else:
#             s1.add(sum%k)
#         cul[sum%k]=i
#     return False
# print(checkSubarraySum([0,0],6))
# def compress(chars):
#     chars.append('*.*')
#     start_zhen=0
#     end_zhen=start_zhen+1
#     while end_zhen<len(chars):
#         if chars[start_zhen]==chars[end_zhen]:
#             end_zhen+=1
#         else:
#             if end_zhen-start_zhen==1:
#                 start_zhen=end_zhen
#                 end_zhen+=1
#             else:
#                 s=str(end_zhen-start_zhen)
#                 for i in range(start_zhen+1,end_zhen):
#                     if i-(start_zhen+1)<len(s):
#                         chars[i]=s[i-(start_zhen+1)]
#                     else:
#                         chars[i]='*.*'
#                 start_zhen=end_zhen
#                 end_zhen+=1
#     for k in range(chars.count('*.*')):
#         chars.remove('*.*')
#     return chars
# print(compress(chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]))
# def wordPattern(pattern: str, s: str) -> bool:
#     ###先将str放入列表
#     res = []
#     start_zhen = 0
#     end_zhen = 1
#     while end_zhen < len(s):
#         if s[end_zhen] != ' ':
#             end_zhen += 1
#         else:
#             res.append(s[start_zhen:end_zhen])
#             start_zhen = end_zhen + 1
#             end_zhen = start_zhen + 1
#     res.append(s[start_zhen:end_zhen+1])
#     if len(s)!=len(res):
#         return False
#     else:
#         dic_c={}
#         s2=set()##象
#         for i in range(len(pattern)):
#             if pattern[i] not in dic_c:
#                 if res[i] in s2:
#                     return False
#                 else:
#                     s2.add(res[i])
#                     dic_c[pattern[i]]=res[i]
#             else:
#                 if dic_c[pattern[i]]!=res[i]:
#                     return False
#         return True
# # print(wordPattern("abba","dog cat cat dog"))
# def lengthOfLongestSubstring(s):
#     dic_c={s[0]}
#     start_zhen=0
#     end_zhen=1
#     res=0
#     while end_zhen<len(s):
#         if s[end_zhen] not in dic_c:
#             dic_c.add(s[end_zhen])
#             end_zhen+=1
#         else:
#             res=max(res,end_zhen-start_zhen)
#             start_zhen+=1
#             end_zhen=start_zhen+1
#             dic_c={s[start_zhen]}
#     return max(res,end_zhen-start_zhen)
# print(lengthOfLongestSubstring(s = "a"))pr
import copy
s=[1,2,3,4]
s1=s
s2=copy.deepcopy(s)
print(id(s))
print(id(s1))
print(id(s2))