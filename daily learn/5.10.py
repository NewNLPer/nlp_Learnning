# -*- coding:utf-8 -*-
# @Time      :2022/5/10 9:58
# @Author    :Riemanner
# Write code with comments !!!
# def uniquePaths(m,n):
#     s = {(1,1): 0, (1,2): 1, (2,1): 1, (2,2): 2}
#     def dfs(tum, s):
#         if tum in s:
#             return s[tum]
#         elif tum[0]==1 or tum[1]==1:
#             return 1
#         else:
#             s[tum] = dfs((tum[0]-1,tum[1]),s) + dfs((tum[0],tum[1]-1),s)
#             return s[tum]
#     return dfs((m,n),s)
# print(uniquePaths(3,3))
# def uniquePathsWithObstacles(obstacleGrid):
#     path, res = [], [0]
#     m, n = len(obstacleGrid), len(obstacleGrid[0])######获得矩阵的长度与宽度
#     def dfs(i, j, path, m, n):
#         if (i, j) == (m-1, n-1) and obstacleGrid[i][j]==0:
#             res[0]+=1
#             return
#         elif i>=m or j>=n:
#             return
#         elif obstacleGrid[i][j]==0:
#             path.append("#")
#             dfs(i+1, j, path, m, n)
#             dfs(i, j+1, path, m, n)
#         else:
#             return 0
#     dfs(0, 0, path,m, n)
#     return res[0]
# print(uniquePathsWithObstacles(obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]))
# class Solution:
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         m, n = len(obstacleGrid), len(obstacleGrid[0])
#         memo = [[-1] * n for _ in range(m)]
#         return self.dfs(obstacleGrid, 0, 0, m, n, memo)
#
#     def dfs(self, grid, i, j, m, n, memo):
#         if grid[i][j] == 1:#######表示这条路不通，计数为零
#             return 0
#         if i == m - 1 and j == n - 1:#####到达终点，然后计数为1
#             return 1
#         if memo[i][j] != -1:#####记忆查询
#             return memo[i][j]
#         if i == m - 1:####边界处理
#             memo[i][j] = self.dfs(grid, i, j + 1, m, n, memo)
#         elif j == n - 1:######边界处理
#             memo[i][j] = self.dfs(grid, i + 1, j, m, n, memo)
#         else:#####大体情况
#             memo[i][j] = self.dfs(grid, i + 1, j, m, n, memo) + self.dfs(grid, i, j + 1, m, n, memo)
#         return memo[i][j]
# print(Solution().uniquePathsWithObstacles(obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]))
# def minPathSum(grid):
#     n = len(grid)  ###行
#     m = len(grid[0])  ###列
#     if n == 1:
#         return sum(grid[0])
#     if m == 1:
#         s = 0
#         for i in range(n):
#             s = s + sum(grid[i])
#         return s
# import copy
# def p(x):
#     x.pop()
#     return x
# def minPathSum(grid):
#     n = len(grid)  ###行
#     m = len(grid[0])  ###列
#     memo={}
#     def dps(n,m,memo,grid):
#         if memo.get((n,m),0)!=0:
#             print('cha')
#             return memo[(n,m)]
#         elif n == 1:
#             return sum(grid[0])
#         elif m == 1:
#             s = 0
#             for i in range(n):
#                 s = s + grid[0][0]
#             return s
#         elif n == 2 and m==2:
#             return min(sum(grid[0]) + grid[-1][-1], grid[0][0] + sum(grid[1]))
#         else:
#             grid1=copy.deepcopy(grid)
#             grid2=copy.deepcopy(grid)
#             grid2=list(map(p,grid2))
#             grid1=grid1[:len(grid1)-1]
#             memo[(n,m)]=min(dps(n-1,m,memo,grid1),dps(n,m-1,memo,grid2))+grid[-1][-1]
#             return memo[(n,m)]
#     return dps(n,m,memo,grid)
# print(minPathSum(grid = [[1,3,1,7],[1,5,1,9],[4,2,1,3]]))
# def summ(list1,list2):
#     for i in range(len(list1)):
#         list1[i]+=list2[i]
#     return list1
# def largestRectangleArea(heights):
#     heights.insert(0, 0)  # 数组头部加入元素0
#     heights.append(0)  # 数组尾部加入元素0
#     st = [0]#单调栈开始进行
#     result = 0
#     for i in range(1, len(heights)):
#         while st != [] and heights[i] < heights[st[-1]]:
#             midh = heights[st[-1]]
#             st.pop()
#             if st != []:
#                 minrightindex = i
#                 minleftindex = st[-1]
#                 summ = (minrightindex - minleftindex - 1) * midh
#                 result = max(summ, result)
#         st.append(i)
#     return result
# def p(x):
#     return int(x)
# def maximalRectangle(matrix):
#     if len(matrix)==0:
#         return 0
#     else:
# ######先对矩阵进行数值处理
#         res=0
#         matrix[0]=list(map(p,matrix[0]))
#         for i in range(1,len(matrix)):
#             matrix[i]=list(map(p,matrix[i]))
#             matrix[i]=summ(matrix[i],matrix[i-1])
#         for j in range(len(matrix)):
#             res=max(res,largestRectangleArea(matrix[j]))
#         return res
# print(maximalRectangle([["0","1"],["1","0"]]))
from functools import lru_cache
@lru_cache()
def isUgly(n):
    if n == 1:
        return True
    elif n == 2:
        return True
    elif n == 3:
        return True
    elif n == 5:
        return True
    else:
        if n % 2 == 0 and isUgly(n // 2):
            return True
        elif n % 3 == 0 and isUgly(n // 3):
            return True
        elif n % 5 == 0 and isUgly(n // 5):
            return True
        else:
            return False
print(isUgly(-2147483648))