# -*- coding:utf-8 -*-
# @Time      :2022/5/20 11:43
# @Author    :Riemanner
# Write code with comments !!!
# class Treenode():
#     def __init__(self,val):
#         self.val=val
#         self.right=None
#         self.left=None
#
# a=Treenode(1)
# b=Treenode(2)
# c=Treenode(3)
# d=Treenode(4)
# e=Treenode(5)
# a.left=b
# b.left=c
# c.left=d
# d.left=e
# # a.right=c
# def maxPathSum(root):
#     res=root.val
#     if not root.right and not root.left:
#         pass
#     elif not root.right and root.left:
#         root.val=max(res,root.val,root.left.val,root.val+root.left.val)
#         maxPathSum(root.left)
#     elif not root.left and root.right:
#         root.val = max(res, root.val,root.right.val,root.val+root.right.val)
#         maxPathSum(root.right)
#     else:
#         root.val = max(res,root.val,root.left.val,root.right.val,root.val+root.left.val,root.val+root.right.val,root.val+root.left.val+root.right.val,maxPathSum(root.left),maxPathSum(root.right))
#     return root.val
# # print(maxPathSum(a))
# def facbo(n):
#     dic_c={1:1,2:1}
#     def bt(n,hash):
#         if n in hash:
#             return hash[n]
#         else:
#             hash[n]=bt(n-1,hash)+bt(n-2,hash)
#             return hash[n]
#     return bt(n,dic_c)
# print(facbo(100))
# def longestPalindrome(s):
#     dic_c = {}
#     for i in s:
#         if i in dic_c:
#             dic_c[i] += 1
#         else:
#             dic_c[i] = 1
#     ji = 0
#     ou = 0
#     for keys in dic_c:
#         if dic_c[keys] % 2 == 0:
#             ou += dic_c[keys]
#         else:
#             ji+=dic_c[keys]-1
#     return ou + ji+1
# # print(longestPalindrome())
# def maxProfit(prices):
#     dic_c = {}
#     n = len(prices)
#     res = 0
#     for i in range(n - 1):  ###买
#         if prices[i] >= prices[i + 1]:
#             continue
#         for j in range(i + 1, n):  ###卖
#             print(prices[i])
#             if j in dic_c:
#                 res = max(res, dic_c[j]-prices[i])
#             else:
#                 dic_c[j] = max(prices[j:])
#                 res=max(res, dic_c[j]-prices[i])
#     print(dic_c)
#     return res

