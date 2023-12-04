# -*- coding:utf-8 -*-
# @Time      :2022/5/3 8:47
# @Author    :Riemanner
# Write code with comments !!!
###关于回溯算法的大总结###
#对于回溯算法大致可以分成四类问题，对于这四类问题要及时进行回顾
###一、子集+排列+组合
###标标准准的开始回溯算法的，考虑到下层元素，然后进行元素筛选，加剪枝操作
###二、切回回文字符串经典例题
# def partition(s: str):
#     res=[]
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res.append(path[:])
#         for i in range(startIndex, len(s)):
#             p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             if p == p[::-1]:
#                 path.append(p)  # 是回文子串
#             else:
#                 continue  # 不是回文，跳过
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#     backtrack(s, 0, [])
#     return res
# print(partition('aaaaaaaa'))
#####s试想可以不可以加上记忆化
from functools import lru_cache
# def wordBreak(s,wordDict) -> bool:
#     @lru_cache()
#     def dfs(i):
#         if i >= len(s):
#             return True
#         for j in range(i, len(s)):
#             if s[i:j + 1] in wordDict and dfs(j + 1):
#                 return True
#         return False
#     return dfs(0)
# @lru_cache()
# def diffWaysToCompute(input: str):
#     res = []
#     ops = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y}
#     for indx in range(1, len(input) - 1):
#         if input[indx] in ops.keys():
#             for left in diffWaysToCompute(input[:indx]):
#                 for right in diffWaysToCompute(input[indx + 1:]):
#                     res.append(ops[input[indx]](left, right))
#     if not res:
#         res.append(int(input))
#     return res
# print(diffWaysToCompute('1-2*3+2-25*2+2-3*4+36+3+9-6*5'))
# @lru_cache()
# def integerReplacement(n: int) :
#     if n == 1:
#         return 0
#     elif n == 2:
#         return 1
#     else:
#         ans = 0
#         if n%2!=0:
#             return ans+min(integerReplacement(n-1),integerReplacement(n+1))+1
#         else:
#             return ans+integerReplacement(n//2)+1
# print(integerReplacement(8000000000000000000))
# def jihe(nums):
#     res=[]
#     def bt(path,start):
#         res.append(path[:])
#         for i in range(start,len(nums)):
#             path.append(nums[i])
#             bt(path,i+1)
#             path.pop()
#     bt([],0)
#     return res
# print(jihe(nums = [4, 3, 2, 3, 5, 2, 1]))

# def partition(s):
#     res = []
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res.append(path[:])
#         for i in range(startIndex, len(s)):
#             p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             path.append(p)  # 是回文子串
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#
#     backtrack(s, 0, [])
#     return res
# print(partition([1,2,3,4]))
# def quanpailie(nums):
#     res = []
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
# print(quanpailie([1,2,3]))
# def searchMatrix(matrix,target):
#     if len(matrix)==1:### 就一行
#         return target in matrix[0]
#     else:
#         for i in range(1, len(matrix)):
#             if matrix[i][0] > target:
#                 for j in range(len(matrix[i - 1])):
#                     if matrix[i - 1][0] > target:
#                         return False
#                     elif matrix[i - 1][j] != target:
#                         continue
#                     else:
#                         return True
#                 return False
#             elif matrix[i][0]==target:
#                 return True
#         return target in matrix[-1]
# print(searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,50]],30))
# print(''.join(sorted('BADC')))

# def zifuchuan(s,t):
#     res=[]
#     def bt(path,start,s,t):
#         if start>=len(s) :
#             res.append(path)
#         for i in range(start,len(s)):
#             p1=''.join(sorted(s[start:i+1]))
#             p2=''.join(sorted(t))
#             if p2 in p1:
#                 path.append(s[start:i+1])
#             else:
#                 continue
#             bt(path,i+1,s,t)
#     bt([],0,s,t)
#     return res

# def partition( s):
#     res = []
#     def backtrack(s, startIndex, path):
#         if startIndex >= len(s):  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             res.append(path[:])
#         for i in range(startIndex, len(s)):
#             p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             path.append(p)  # 是回文子串
#             backtrack(s, i + 1, path)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#
#     backtrack(s, 0, [])
#     return res
# print()
# def xunzaho(s,t):
#     for i in range(len(t),len(s)):
#         start_zhen=0
#         end_zhen=i
#         while end_zhen<len(s):
#             s1=[]
#             s2=[]
#             q1=s[start_zhen:end_zhen]
#             s1+=q1
#             s2+=t
#             if set(s1)==set(s1)
# b = []
# c = '1234'
# b += c
# print(b)
# def largestRectangleArea(heights):
#     res = 1
#     for i in range(len(heights) - 1):
#         for j in range(i+1, len(heights)):
#             res =max(heights[j],res,min(heights[i:j+1])*(j-i+1))
#     return res
# print(largestRectangleArea(heights = [2,4]))
class Node():
    def __init__(self,val):
        self.val=val
        self.next=None
a=Node(1)
b=Node(2)
c=Node(2)
d=Node(2)
# e=Node(5)
a.next=b
b.next=c
c.next=d
# d.next=e
def dayin(head):
    while head:
        print(head.val,end=' ')
        head= head.next
dayin(a)
print()


# def reorderList(head):
#     """
#     Do not return anything, modify head in-place instead.
#     """
#     if head.next == None or head.next.next == None:
#         return head
#     else:
#         s=[]
#         head1 = head#####拿到尾部的前一个
#         while head1.next.next:
#             s.append(head1)
#             head1 = head1.next
#         tmp=head.next
#         head.next=head1.next
#         head1.next=None
#         head.next.next=reorderList(tmp)
#         return head
# cc=reorderList(a)
# dayin(cc)
# def removeElements(head,val):
#     if not head:
#         return head
#     else:
#         head1=Node(999)
#         head1.next=head
#         tail=head1
#         while tail.next:
#             if tail.next.val==val:
#                 tail.next = tail.next.next
#             else:
#                 tail = tail.next
#         return head1.next
# cc=removeElements(a,2)
# dayin(cc)
# def isPalindrome(s: str) -> bool:
#     if len(s) == 1:
#         return True
#     else:
#         start_zhen = 0
#         end_zhen = len(s) - 1
#         while start_zhen < end_zhen:
#             if not s[start_zhen].isalpha() and not s[start_zhen].isdigit():
#                 start_zhen += 1
#             elif not s[end_zhen].isalpha() and not s[end_zhen].isdigit():
#                 end_zhen -= 1
#             elif s[start_zhen].lower() == s[end_zhen].lower():
#                 start_zhen += 1
#                 end_zhen -= 1
#             else:
#                 return False
#         return True
# print(isPalindrome("a."))