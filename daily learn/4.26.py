# -*- coding:utf-8 -*-
# @Time      :2022/4/26 11:26
# @Author    :Riemanner
# Write code with comments !!!
####回溯算法及其案例
import copy
###子集问题
# def ziji(nums):
#     def bt(nums,path,start):
#         res.append(copy.deepcopy(path))
#         for i in range(start,len(nums)):###深度
#             if i==start or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 bt(nums,path,i+1)###nums为广度
#                 path.pop()
#     res=[]
#     bt(nums,[],0)
#     return res
# print(ziji([1,2,2]))
# def zuhe2(num,target):
#     def bt(num,path):
#         if sum(path)==target:
#             res.append(copy.deepcopy(path))
#         elif sum(path)>target:
#             return
#         for i in range(len(num)):
#             path.append(num[i])
#             bt(num[i:],path)
#             path.pop()
#     res=[]
#     bt(num,[])
#     return res
# print(zuhe2([2,3,5],8))
# def zuhe(nums):
#     def bt(nums,path):
#         if len(path)==len(nums):
#             res.append(copy.deepcopy(path))
#         for i in range(len(nums)):
#             if nums[i] not in path:
#                 path.append(nums[i])
#                 bt(nums,path)
#                 path.pop()
#     res=[]
#     bt(nums,[])
#     return res
# print(zuhe([2,1]))
#####动态规划对，斐波那契数列与阶乘的计算
###先对列表进行调整n，k
# def factorial_dp(num):
#     if num == 0:
#         return 1
#     dp = [1] * (num + 1)
#     for i in range(1,num+1):
#         dp[i] = dp[i-1]*i
#     return dp[num]
# def tiaozheng(n,k):
#     k=k-1
#     nums=list(range(1,n+1))
#     nums.insert(0,nums[k // factorial_dp(n - 1)])
#     nums=nums[0:k // factorial_dp(n - 1)+1]+nums[k // factorial_dp(n - 1)+2:]
#     s=k-(k // factorial_dp(n - 1))*factorial_dp(n-1)+1
#     def bt(nums,path):
#         if len(path)==len(nums):
#             if len(res)==s:
#                 return
#             else:
#                 res.append(copy.deepcopy(path))
#         for i in range(len(nums)):
#             if nums[i] not in path:
#                 path.append(nums[i])
#                 bt(nums,path)
#                 path.pop()
#             if i!=0 and len(res)==s:
#                 break
#     res=[]
#     bt(nums,[])
#     return "".join(list(map(str,res[-1])))
# print(tiaozheng(4,1))
###采用回溯法进行数独的解读

#
# def dfs(board, n, ret):
#     if n == 81:
#         # 判断棋盘是否合法
#         if isValidSudoku(board):
#             ret = board.copy()
#         return
#     x, y = n / 9, n % 9
#     if board[x][y] != '.':
#         dfs(board, n+1, ret)
#     for i in range(9):
#         c = str(i+1)
#         board[x][y] = c
#         dfs(board, n+1, ret)
#         board[x][y] = '.'

# def solveNQueens(n):
#     def isValid(row, col):
#             # 判断所放置的位置是否合法
#             for i in range(row):
#                 for j in range(n):
#                     if board[i][j] == 'Q' and (j == col or i + j == row + col or i-j == row-col):
#                         return False
#             return True
#     def backtrack(board, row):####寻找回溯的规律，判断过程
#         if row == n:
#             cur_res = [''.join(row) for row in board]
#             print(cur_res)
#             res.append(cur_res)
#             return
#         for i in range(n):
#             if isValid(row, i):
#                 board[row][i] = 'Q'
#                 backtrack(board, row+1)
#                 board[row][i] = '.'
#     res = []
#     board = [['.'] * n for _ in range(n)]
#     backtrack(board,0)
#     return res
# print(solveNQueens(4))


# def isValidSudoku(board,i,j,m):
#     board[i][j]=m
#     for i in range(9):
#         for j in range(9):
#             if board[i][j] == '.':
#                 continue
#             else:
#                 ss = []
#                 for o in range(9):
#                     ss.append(board[o][j])
#                 if board[i].count(board[i][j]) == 1 and ss.count(board[i][j]) == 1:
#                     S = []  ###对九宫格的数据进行储存
#                     for l in range(3 * (i // 3), 3 * (i // 3) + 3):
#                         for k in range(3 * (j // 3), 3 * (j // 3) + 3):
#                             S.append(board[l][k])
#                     if S.count(board[i][j]) != 1:
#                         return False
#                 else:
#                     return False

# class Solution:
#     def solveSudoku(self, board) -> None:
#         """
#         Do not return anything, modify board in-place instead.
#         """
        # def backtrack(board,target):
        #     if target==81:
        #         return
        #     for i in range(len(board)):
        #         for j in range(len(board[0])):
        #             if board[i][j]=='.':
        #                 for k in range(1,10):
        #                     if self.check(board,i,j,str(k)):
        #                         board[i][j]=str(k)
        #                         print(board)
        #                         backtrack(board,target+1)
        #                          board[i][j]='.'
        #
        # backtrack(board,0)
        # return board
#         for i in range(len())
#     def check(self, board, row, col, c):
#         for i in range(9):
#             if board[row][i] == c:
#                 return False
#             if board[i][col] == c:
#                 return False
#             if board[(row//3)*3 + i // 3][(col//3)*3 + i % 3] == c:
#                 return False
#         return True
#
#
# Solution().solveSudoku(board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]])

# class Solution:
#     def solveSudoku(self, board) -> None:
#         """
#         Do not return anything, modify board in-place instead.
#         """
#         if board is None or not len(board):
#             return
#         self.solve(board)
#
#     def solve(self, board):
#         for i in range(len(board)):
#             for j in range(len(board[0])):
#                 if board[i][j] == '.':
#                     for c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
#                         if self.isValid(board, i, j, c):
#                             board[i][j] = c
#                             if self.solve(board):
#                                 return True
#                             else:
#                                 board[i][j] = '.'
#                     return False
#         return True
#
#     def isValid(self, board, row, col, c) -> bool:
#         for i in range(9):
#             if board[row][i] != '.' and board[row][i] == c:
#                 return False
#             if board[i][col] != '.' and board[i][col] == c:
#                 return False
#             if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] != '.' and board[3 * (row // 3) + i // 3][
#                 3 * (col // 3) + i % 3] == c:
#                 return False
#         return True
import copy
# def combinationSum3(k,n):
#     def bt(n, k, path,start):
#         if len(path) == k and sum(path)==n:
#             res.append(copy.deepcopy(path))
#         elif sum(path)>n:
#             return
#         elif len(path)>k:
#             return
#         for i in range(start, 10):
#             if i not in path:
#                 path.append(i)
#                 bt(n, k, path,i+1)
#                 path.pop()
#
#     res = []
#     bt(n, k, [],1)
#     return res
# print(combinationSum3(4,1))
# def partition(s):
#     def bt(s, path, start):
#         if len(path) != 0 and len(path) != s:
#             res.append(copy.deepcopy(path))
#         for i in range(start,len(s)):
#             path.append(s[i])
#             bt(s, path, i+1)
#             path.pop()
#     res=[]
#     bt(s,[],0)
#     return res
# print(partition('abc'))
###一个小时回顾
###链表，三个经典回溯算法，解数独+n皇后
# class Node():
#     def __init__(self,val):
#         self.val=val
#         self.next=None
# a=Node(1)
# b=Node(2)
# c=Node(3)
# d=Node(4)
# a.next=b
# b.next=c
# c.next=d
#
# def dayin(head):
#     while head:
#         print(head.val,end=' ')
#         head=head.next
# def fnahzu(head):
#     pre=None
#     cur=head
#     while cur:
#         tmp=cur.next
#         cur.next=pre
#         pre=cur
#         cur=tmp
#     return pre
# dayin(a)
# cc=fnahzu(a)
# print()
# dayin(cc)
####关于子集的问题
# def ziji(nums,target):
#     def bt(path,nums):
#         if sum(path)==target:
#             res.append(copy.deepcopy(path))
#         elif sum(path)>target:
#             return
#         for i in range(len(nums)):
#             path.append(nums[i])
#             bt(path,nums[i:])
#             path.pop()
#     res=[]
#     bt([],nums)
#     return res
# print(ziji([1,2,3],6))


# class Solution:
#     def solveSudoku(self, board: List[List[str]]) -> None:
#         """
#         Do not return anything, modify board in-place instead.
#         """
#         if board is None or not len(board):
#             return
#         self.solve(board)
#
#     def solve(self, board):
#         for i in range(len(board)):
#             for j in range(len(board[0])):
#                 if board[i][j] == '.':
#                     for c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
#                         if self.isValid(board, i, j, c):
#                             board[i][j] = c
#                             if self.solve(board):
#                                 return True
#                             else:
#                                 board[i][j] = '.'
#                     return False
#         return True
#
#     def isValid(self, board, row, col, c) -> bool:
#         for i in range(9):
#             if board[row][i] != '.' and board[row][i] == c:
#                 return False
#             if board[i][col] != '.' and board[i][col] == c:
#                 return False
#             if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] != '.' and board[3 * (row // 3) + i // 3][
#                 3 * (col // 3) + i % 3] == c:
#                 return False
#         return True
# def findSubsequences(nums):
#     def bt(start, path, nums):
#         if len(path) >= 2 and sorted(path)==path:
#             res.append(copy.deepcopy(path))
#         for i in range(start, len(nums)):
#             if i ==start or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 bt(i + 1, path, nums)
#                 path.pop()
#
#     res = []
#     bt(0, [], nums)
#     return res
# print(findSubsequences([4,6,7,7]))
def permutation(s):
    def bt(s, path):
        if not s:
            res.append(copy.deepcopy(path))
        for i in range(len(s)):
            if i ==0 or s[i]!=s[i-1]:
                path = path + s[i]
                bt(s[:i]+s[i+1:], path)
                path = path[0:len(path )- 1]
    res = []
    bt(s, '')
    return res
print(permutation('aab'))


["suvyls","suvysl","suvlys","suvlsy","suvsyl","suvsly","suyvls","suyvsl","suylvs","suylsv","suysvl","suyslv","sulvys","sulvsy","sulyvs","sulysv","sulsvy","sulsyv","susvyl","susvly","susyvl","susylv","suslvy","suslyv","svuyls","svuysl","svulys","svulsy","svusyl","svusly","svyuls","svyusl","svylus","svylsu","svysul","svyslu","svluys","svlusy","svlyus","svlysu","svlsuy","svlsyu","svsuyl","svsuly","svsyul","svsylu","svsluy","svslyu","syuvls","syuvsl","syulvs","syulsv","syusvl","syuslv","syvuls","syvusl","syvlus","syvlsu","syvsul","syvslu","syluvs","sylusv","sylvus","sylvsu","sylsuv","sylsvu","sysuvl","sysulv","sysvul","sysvlu","sysluv","syslvu","sluvys","sluvsy","sluyvs","sluysv","slusvy","slusyv","slvuys","slvusy","slvyus","slvysu","slvsuy","slvsyu","slyuvs","slyusv","slyvus","slyvsu","slysuv","slysvu","slsuvy","slsuyv","slsvuy","slsvyu","slsyuv","slsyvu","ssuvyl","ssuvly","ssuyvl","ssuylv","ssulvy","ssulyv","ssvuyl","ssvuly","ssvyul","ssvylu","ssvluy","ssvlyu","ssyuvl","ssyulv","ssyvul",...