# -*- coding:utf-8 -*-
# @Time      :2022/5/13 9:43
# @Author    :Riemanner
# Write code with comments !!
# def solve(board):
#     row, col = len(board), len(board[0])
#     def dfs(x, y):
#         if board[x][y] != 'O':
#             return
#         else:
#             board[x][y] = '#'
#         for c in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
#             if 0 <= x + c[0] < row and 0 <= y + c[1] < col:
#                 dfs(x + c[0], y + c[1])
#
#     for i in range(row):
#         dfs(i, 0)
#         dfs(i, col - 1)
#     for j in range(1, col - 1):
#         dfs(0, j)
#         dfs(row - 1, j)
#
#     for i in range(row):
#         for j in range(col):
#             board[i][j] = 'O' if board[i][j] == '#' else 'X'
#     return board
# print(solve(board = [["X","O","O","O",'X'],["X","X","O","X",'X'],["X","X","O","X",'X'],['X',"X","O","X","X"],["X","X","O","X",'X']]))

# def solve2(board):
#     row, col = len(board), len(board[0])
#     k=0
#     def dfs(x, y):
#         if board[x][y] == '0' or board[x][y]=='#':
#             return
#         else:
#             board[x][y] = '#'
#         for c in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
#             if 0 <= x + c[0] < row and 0 <= y + c[1] < col:
#                 dfs(x + c[0], y + c[1])
#     for i in range(row):
#         for j in range(col):
#             if board[i][j]=='#' or board[i][j]=='0':
#                 continue
#             elif board[i][j]=='1':
#                 k+=1
#                 dfs(i,j)
#     return k
# print(solve2([
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]))
# def isHappy(n):
#     while True:
#         s = str(n)
#         q = 0
#         for i in s:
#             q += pow(int(i), 2)
#         if q == 1:
#             return True
#         elif q == 4:
#             return False
#         else:
#             n = q
# print(isHappy(19))
def computeArea(ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
    ####存在重合，无重合
    s1 = [ax1, ax2, bx1, bx2]
    s2 = [ay1, ay2, by1, by2]
    s1.sort()
    s2.sort()
    print(max((s1[2] - s1[1]) * (s2[2] - s2[1]), 0))
    return abs(ax2-ax1)*abs(ay2-ay1)+abs(bx2-bx1)*abs(by2-by1)-max((s1[2] - s1[1]) * (s2[2] - s2[1]), 0)
print(computeArea(-2,-2,2,2,3,3,4,4))