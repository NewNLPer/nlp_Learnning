# -*- coding:utf-8 -*-
# @Time      :2022/4/27 11:05
# @Author    :Riemanner
# Write code with comments !!!


# dirs=[
#     lambda x,y:(abs(x),abs(y+1)),
#     lambda x,y:(abs(x+1),abs(y)),
#     lambda x,y:(abs(x),abs(y-1)),
#     lambda x,y:(abs(x-1),abs(y))
# ]
# def maze_path(x1,y1,board,word):##找到起点开始进行判断
#     sss=[]
#     for i in range(len(board)):
#         s=[]
#         for j in range(len(board[0])):
#             s.append(0)
#         sss.append(s)
#     sss[x1][y1]=1
#     ss=[board[x1][y1]]####对比word
#     stack=[(x1,y1)]###记录路径
#     o=1
#     while len(stack)>0:
#        curNode=stack[-1]
#        if len(ss)==len(word) and ''.join(ss)==word:
#            return True
#        for dir in dirs:
#            nextNode=dir(curNode[0],curNode[1])
#            if nextNode[0]<=len(board)-1 and nextNode[1]<=len(board[0])-1 and board[nextNode[0]][nextNode[1]]==word[o] and sss[nextNode[0]][nextNode[1]]==0:#在范围内而且等于
#                 ss.append(board[nextNode[0]][nextNode[1]])
#                 stack.append(nextNode)
#                 sss[nextNode[0]][nextNode[1]]=1#表示已经走过：
#                 o+=1
#                 break
#        else:
#            stack.pop()
#            sss[curNode[0]][curNode[1]]=2
#            ss.pop()
#            o-=1
#
#     else:
#         return False
# def exist(board,word):
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             if board[i][j]!=word[0]:
#                 continue
#             elif maze_path(i,j,board,word)==False:
#                 continue
#             elif maze_path(i,j,board,word)==True:
#                 return True
#     return False
# print(exist([["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]],"ABCEFSADEESE"))
# def solveSudoku(board):
#     """
#     Do not return anything, modify board in-place instead.
#     """
#     if board is None or not len(board):
#         return
#     self.solve(board)
#
#
import copy


def solve(self, board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == '.':
                for c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    if self.isValid(board, i, j, c):
                        board[i][j] = c
                        if self.solve(board):
                            return True
                        else:
                            board[i][j] = '.'
                return False
    return True
#
#
# def isValid(self, board, row, col, c) -> bool:
#     for i in range(9):
#         if board[row][i] != '.' and board[row][i] == c:
#             return False
#         if board[i][col] != '.' and board[i][col] == c:
#             return False
#         if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] != '.' and board[3 * (row // 3) + i // 3][
#             3 * (col // 3) + i % 3] == c:
#             return False
#     return True
# def exist(board,word) -> bool:
#     n = len(board)
#     m = len(board[0])
#     len_str = len(word)
#
#     def backtracking(i, j, p):
#         if p == len_str: return True
#         if i < 0 or i >= n or j < 0 or j >= m: return False
#         if word[p] == board[i][j]:
#             board[i][j] = '0'
#             if backtracking(i - 1, j, p + 1) or backtracking(i + 1, j, p + 1) or backtracking(i, j - 1,
#                                                                                               p + 1) or backtracking(i,
#                                                                                                                      j + 1,
#                                                                                                                      p + 1): return True
#             board[i][j] = word[p]
#         print(board)
#         return False
#
#     for i in range(n):
#         for j in range(m):
#             if board[i][j] == word[0]:
#                 if backtracking(i, j, 0):
#                     print(board)
#                     return True
#     return False
# print(exist(board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"))
# def partition(s):
#     res = []
#     path = []  # 放已经回文的子串
#
#     def backtrack(s, startIndex):
#         if startIndex >= len(s):  # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
#             return res.append(path[:])
#         for i in range(startIndex, len(s)):
#             p = s[startIndex:i + 1]  # 获取[startIndex,i+1]在s中的子串
#             if p == p[::-1]:
#                 path.append(p)  # 是回文子串
#             else:
#                 continue  # 不是回文，跳过
#             backtrack(s, i + 1)  # 寻找i+1为起始位置的子串
#             path.pop()  # 回溯过程，弹出本次已经填在path的子串
#
#     backtrack(s, 0)
#     return res
# print(partition('aabc'))    def combinationSum4(self, nums: List[int], target: int) -> int:
def combinationSum4(nums,target):
    def bt(nums, path, target):
        if sum(path) == target:
            res[0]+=1
        elif sum(path)>target:
            return
        for i in range(len(nums)):
            path.append(nums[i])
            bt(nums, path, target)
            path.pop()
    res=[0]
    bt(nums, [], target)
    return res[0]
print(combinationSum4([1,2,3],4))