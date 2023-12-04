# -*- coding:utf-8 -*-
# @Time      :2022/5/17 8:04
# @Author    :Riemanner
# Write code with comments !!!
# class Bitreed:
#     def __init__(self,val):
#         self.val=val
#         self.left=None
#         self.right=None
# A=Bitreed(1)
# B=Bitreed(2)
# C=Bitreed(2)
# D=Bitreed(2)
# E=Bitreed(2)
# # F=Bitreed(6)
#
# # D=Bitreed('D')
# # E=Bitreed('E')
# # F=Bitreed('F')
# # G=Bitreed('G')
#
# A.left=B
# A.right=C
# B.left=D
# C.left=E
# # print(A.lchild.data)
# # root=E
# # A.lchild=B
# # # B.lchild=C
# root=A
####二叉树的遍历，分为前序遍历，中序遍历，后序遍历，还有层次遍历
###前序遍历
# def pre_order(root):
#     if root:
#         print(root.data,end=' ')
#         pre_order(root.lchild)
#         pre_order(root.rchild)
# pre_order(root)
# ###中序遍历
# print()

# def mid_order(root):
#     if root:
#         mid_order(root.lchild)
#         print(root.data,end=' ')
#         mid_order(root.rchild)
# mid_order(root)
# print()
###后序遍历
# def hou_order(root):
#     if root:
#         hou_order(root.lchild)
#         hou_order(root.rchild)
#         print(root.data,end=' ')
# hou_order(root)
# ###层次遍历
# print()
# from collections import deque
# def level_order(root):
#     queue=deque()
#     queue.append(root)
#     while queue:####使得队列不会空
#         node=queue.popleft()
#         print(node.data,end=' ')
#         if node.lchild:
#             queue.append(node.lchild)
#         if node.rchild:
#             queue.append(node.rchild)
# level_order(root)
# print(root.left.val)
# print(root.right.val)
# def isSameTree(p,q):
#     if not p and not q:
#         return True
#     elif p and q:
#         if p.val==q.val:
#             if isSameTree(p.left,q.left) and isSameTree(p.right,q.right):
#                 return True
#             else:
#                 return False
#         else:
#             return False
#     else:
#         return False
# def isSymmetric(root):
#     if not root.left and not root.right:
#         return True
#     elif (root.left and not root.right) or (not root.left and root.right):
#         return False
#     elif root.left and root.right and root.right.val!=root.left.val:
#         return False
#     elif root.left and root.right and root.right.val==root.left.val and not root.left.right and not root.left.left and not root.right.right and not root.right.left:
#         return True
#     else:
#         return isSameTree(root.left.left,root.right.right) and isSameTree(root.left.right,root.right.left)
# print(isSymmetric(root))
# from collections import deque
# def level_order(root):
#     queue=deque()
#     queue.append(root)
#     s=[]
#     while queue:####使得队列不会空
#         node=queue.popleft()
#         s.append(node.val)
#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)
#     return s
# print(level_order(root))
# def isSymmetric(root):
#     queue = [root]
#     while (queue):
#         next_queue = list()
#         layer = list()
#         for node in queue:
#             if not node:
#                 layer.append(None)
#                 continue
#             next_queue.append(node.left)
#             next_queue.append(node.right)
#             layer.append(node.val)
#         if layer != layer[::-1]:
#             return False
#         queue = next_queue
#     return True
# print(isSymmetric(root))
#######dfs的复习
#无重复子集
# def ziji1(nums):
#     res=[]
#     def bt(nums,start,path):
#         res.append(path[:])
#         for i in range(start,len(nums)):
#             path.append(nums[i])
#             bt(nums,i+1,path)
#             path.pop()
#     bt(nums,0,[])
#     return res
# print(ziji1([1,2,3]))
#有重复子集
# def ziji2(nums):
#     res=[]
#     def bt(nums,start,path):
#         res.append(path[:])
#         for i in range(start,len(nums)):
#             if i==0 or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 bt(nums,i+1,path)
#                 path.pop()
#     bt(nums,0,[])
#     return res
# print(ziji2([1,2,2]))
#无重复k组合
#有重复k组合
#####其实也是组合问题，就是对子集的数量进行了限制罢了，对于res.append(path[:])加上限制条件既可以，不再重复进行说明
#无重复排列
# def pailie1(nums):
#     res=[]
#     def bt(nums,path):
#         if len(path)==len(nums):
#             res.append(path[:])
#             return
#         for i in range(len(nums)):
#             if nums[i] not in path:
#                 path.append(nums[i])
#                 bt(nums,path)
#                 path.pop()
#     bt(nums,[])
#     return res
# print(pailie1([1,1,3]))
#有重复排列
# def palie2(nums):
#     res=[]
#     n=len(nums)
#     def br(nums,path):
#         if not nums:
#             res.append(path[:])
#             return
#         for i in range(len(nums)):
#             if i==0 or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 br(nums[:i]+nums[i+1:],path)
#                 path.pop()
#     br(nums,[])
#     return res
# print(palie2([1,1,2]))
####切割回文字符串，
# def huiwen(s):
#     res=[]
#     def bt(s,path,start):
#         if start>=len(s):
#             res.append(path[:])
#             return
#         for i in range(start,len(s)):
#             p=s[start:i+1]
#             if p==p[::-1]:
#                 path.append(p)
#             else:
#                 continue
#             bt(s, path, i + 1)
#             path.pop()
#     bt(s,[],0)
#     return res
# print(huiwen('ada'))
# class Bitreed:
#     def __init__(self,val):
#         self.val=val
#         self.left=None
#         self.right=None
# A=Bitreed(1)
# B=Bitreed(2)
# C=Bitreed(3)
# D=Bitreed(4)
# E=Bitreed(5)
#
# A.left=B
# A.right=C
# B.left=D
# C.left=E

# def maxDepth(root):
#     if not root:
#         return 0
#     else:
#         return max(maxDepth(root.right),maxDepth(root.left))+1
# def isBalanced(root):
#     if not root:
#         return True
#     elif (not root.right and not root.left) or ((not root.right and root.left)) or (root.right and not root.left):
#         return True
#     else:
#         if abs(maxDepth(root.left)-maxDepth(root.right))<=1:
#             return True
#         else:
#             return False

# class Queue(object) :
#     def __init__(self, size):
#         self.size = size
#         self.queue = []
#     def add(self,val):
#         self.queue.append(val)
#         return self.queue
#     def pop(self):
#         self.queue.pop(0)
#         return self.queue
# c= Queue(5)
# print(c.size)

# def exist(board,word):
#     n=len(board)
#     m=len(board[0])
#     p1=len(word)
#     def bt(i,j,p):
#         if p==p1:
#             return True
#         if i>=n or i<0 or j>=m or j<0:
#             return False
#         if board[i][j]==word[p]:
#             board[i][j]='0%s'%board[i][j]
#             if bt(i+1,j,p+1) or bt(i-1,j,p+1) or bt(i,j+1,p+1) or bt(i,j-1,p+1):
#                 return True
#             board[i][j]=word[p]
#         else:
#             return False
#     for i in range(n):
#         for j in range(m):
#             if board[i][j]==word[0]:
#                 if bt(i,j,0):
#                     print(board)
#                     return True
#     print(board)
#     return False
# def findWords(board,words):
#     s=[]
#     for i in words:
#         if exist(board,i):
#             s.append(i)
#             p1=0
#             for j in range(len(board)):
#                 for k in range(len(board[0])):
#                     if len(board[j][k])==2:
#                         board[j][k]=board[j][k][-1]
#                         p1+=1
#     return s
class Solution:
    def findWords(self, board: [[str]], words: [str]) -> [str]:
        m, n = len(board), len(board[0])
        res = set()

        @lru_cache(None)
        def dfs(x, y, ans, mark=set()):
            mark.add((x, y))

            if ans in words:
                res.add(ans)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= x + dx < m and 0 <= y + dy < n and (x + dx, y + dy) not in mark:
                    dfs(x + dx, y + dy, ans + board[x + dx][y + dy])
            mark.remove((x, y))

        for i in range(m):
            for j in range(n):
                dfs(i, j, board[i][j])
        return list(res)

