# -*- coding:utf-8 -*-
# @Time      :2022/6/2 9:47
# @Author    :Riemanner
# Write code with comments !!!
# import copy
# def gameOfLife(board):
#     """
#     Do not return anything, modify board in-place instead.
#     """
#     boards=copy.deepcopy(board)
#     dirs=[lambda x,y:(x+1,y),
#     lambda x,y:(x-1,y),
#     lambda x,y:(x,y-1),
#     lambda x,y:(x,y+1),
#     lambda x,y:(x-1,y-1),
#     lambda x,y:(x-1,y+1),
#     lambda x,y:(x+1,y-1),
#     lambda x,y:(x+1,y+1)
#     ]
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             res=[]
#             for dir in dirs:
#                 node=dir(i,j)
#                 if node[0]>=0 and node[0]<len(board) and node[1]>=0 and node[1]<len(board[0]):
#                     res.append(boards[node[0]][node[1]])
#             print(res)
#             if board[i][j]==1:
#                 if res.count(1)<2 or res.count(1)>3:
#                     board[i][j]=0
#             elif board[i][j]==0:
#                 if res.count(1)==3:
#                     board[i][j]=1
#     return board
# print(gameOfLife([[0,1,0],[0,0,1],[1,1,1],[0,0,0]]))
# def coinChange(coins,amount):
#     memo = {0: 0}
#     def helper(n):
#         if n in memo:
#             return memo[n]
#         res = float("inf")
#         for coin in coins:
#             if n >= coin:
#                 res = min(res, helper(n - coin) + 1)
#         memo[n] = res
#         return res
#     return helper(amount) if (helper(amount) != float("inf")) else -1
# print(coinChange([186,419,83,408],6249))
def change(amount,coins):
    def dfs(amount,coins,memo):
        if amount<0:
            return 0
        if amount==0:
            return 1
        if amount in memo.keys():
            return memo[amount]
        ans = 0
        for i in range(len(coins)):
            ans += dfs(amount - coins[i], coins[i:], memo)
        memo[amount] = ans
        return ans
    memo = {}
    cc=dfs(amount, coins, memo)
    return cc
print(change(amount = 5, coins = [1, 2, 5]))