#np。array,要保证维数一样，然后再进行调选
# import numpy as np
# A = np.array([1,1,1])
# B = np.array([2,2,0])
# c=np.vstack((A,B))
# print(c)
#####np.array 和合并问题解决
##
# def convert( s, numRows):
#     s1 = ''
#     if len(s) <= numRows:
#         return s
#     elif len(s) == 2:
#         for i in range(0, len(s), 2):
#             s1 = s1 + s[i]
#         for j in range(1, len(s), 2):
#             s1 = s1 + s[j]
#         return s1
#     else:
#         S = [''] * numRows
#         n = 2 * (numRows - 1)
#         for i in range(len(s)):
#             c = min(i % n, n - (i % n))
#             S[c] = S[c] + s[i]
#     return ''.join(S)
# print(convert('ABC',1))
# def searchRange(nums,target):
#     if len(nums) == 0:
#         return [-1, -1]
#     elif nums[0] > target:
#         return [-1, -1]
#     elif nums[-1] < target:
#         return [-1, -1]
#     else:
#         S = []
#         n = len(nums) // 2
#         if nums[n] == target:
#             for i in range(n, -1, -1):
#                 if nums[i] == target:
#                     S.append(i)
#             for j in range(n + 1, len(nums)):
#                 if nums[j] == target:
#                     S.append(j)
#             if len(S) == 1:
#                 return [S[0], S[0]]
#             else:
#                 return sorted(S)
#         elif nums[n] > target:
#             for i in range(n - 1, -1, -1):
#                 if nums[i] == target:
#                     S.append(i)
#             if len(S) == 0:
#                 return [-1, -1]
#             else:
#                 return sorted(S)
#         elif nums[n] < target:
#             for j in range(n + 1, len(nums)):
#                 if nums[j] == target:
#                     S.append(j)
#             if len(S) == 0:
#                 return [-1, -1]
#             else:
#                 return sorted(S)
# print(searchRange([1,3],1))
#在考虑二叉树的遍历，三种，还有对于二叉树的一系列操作
# class BiTreeNode:#定义二叉树的结点
#     def __init__(self,data):
#         self.data=data
#         self.lchild=None
#         self.rchild=None
#
# a=BiTreeNode('A')
# b=BiTreeNode('B')
# c=BiTreeNode('C')
# d=BiTreeNode('D')
# e=BiTreeNode('E')
# f=BiTreeNode('F')
# g=BiTreeNode('G')
# e.lchild=a
# e.rchild=g
# a.rchild=c
# c.lchild=b
# c.rchild=d
# g.rchild=f
# root=e
# # print(root.lchild.rchild.data)
# #二叉树的遍历
# #前序遍历
# def fun1(root):
#     if root:
#         print(root.data,end=' ')
#         fun1(root.lchild)
#         fun1(root.rchild)
# fun1(root)
# #中序遍历
# #后序遍历
#二叉搜索树
# class BiTreeNode:#定义二叉树的结点
#     def __init__(self,data):
#         self.data=data
#         self.lchild=None
#         self.rchild=None
#         self.parent=None
# class BST:
#     def __init__(self):
#         self.root=None
#
#     def insert(self,node,val):
#         if not node:
#             node=BiTreeNode(val)
#         else:
#         return node
#

# def longestPalindrome(s) :
#     S=[]
#     if len(s) == 1:
#         return s
#     if len(s)==2 and s[0]!=s[1]:
#         return s[0]
#     else:
#         for i in range(len(s)):
#             for j in range(len(s) - 1, i, -1):
#                 if s[i] != s[j]:
#                     continue
#                 if s[i] == s[j]:
#                     if s[i:j+1]==s[i:j+1][::-1]:
#                         S.append(s[i:j+1])
#                     else:
#                         continue
#         res = max(S, key=len, default='')
#         return res
#
# print(longestPalindrome('aacabDkacaa'))


