# ##########f
# def isValidSudoku(board):
#     for i in range(9):
#         for j in range(9):
#             if board[i][j] == '':
#                 continue
#             else:
#                 if board[i].count(board[i][j]) == 1 and board[j].count(board[i][j]) == 1:
#                     S = []  ###对九宫格的数据进行储存
#                     if i // 3 == i / 3 and j // 3 == j / 3:
#                         for l in range(3 * (i // 3) - 2, (3 * (i // 3)) + 1):
#                             for k in range(3 * (j // 3) - 2, (3 * (j // 3)) + 1):
#                                 S.append(board[l][k])
#                             print(S)
#                     elif i // 3 < i / 3 and j // 3 < j / 3:
#                         for l in range(3 * (i // 3) + 1, 3 * (i // 3) + 3):
#                             for k in range(3 * (j // 3) + 1, 3 * (j // 3) + 3):
#                                 S.append(board[l][k])
#                             print(S)
#                     elif i // 3 == i / 3 and j // 3 < j / 3:
#                         for l in range(3 * (i // 3) - 2, 3 * (i // 3) + 1):
#                             for k in range(3 * (j // 3) + 1, 3 * (j // 3) + 3):
#                                 S.append(board[l][k])
#                             print(S)
#                     elif i // 3 < i / 3 and j // 3 == j / 3:
#                         for l in range(3 * (i // 3) + 1, 3 * (i // 3) + 3):
#                             for k in range(3 * (j // 3) - 2, 3 * (j // 3) + 1):
#                                 S.append(board[l][k])
#                             print(S)
#                     if S.count(board[i][j]) != 1:
#                         return False
#                 else:
#                     return False
#     return True
# print(isValidSudoku(
# def isValidSudoku(board) -> bool:
    # for i in range(9):
    #     for j in range(9):
    #         if board[i][j] == '.':
    #             continue
    #         else:
    #             ss = []
    #             for o in range(9):
    #                 ss.append(board[o][j])
    #             if board[i].count(board[i][j]) == 1 and ss.count(board[i][j]) == 1:
    #                 S = []  ###对九宫格的数据进行储存
    #                 for l in range(3 * (i // 3), 3 * (i // 3) + 3):
    #                     for k in range(3 * (j // 3), 3 * (j // 3) + 3):
    #                         S.append(board[l][k])
    #                 if S.count(board[i][j]) != 1:
    #                     return False
    #             else:
    #                 return False
    # return True
# s=[[1,2],[4,5],[2,2],[0,8]]
# s.sort(key=lambda x:x[0])
# print(s)
def merge(intervals):
    if len(intervals) == 1:
        return intervals
    else:
        s=[]
        i=1
        intervals.sort(key=lambda x: x[0])
        while i<len(intervals):
            if intervals[i][0]<= intervals[i-1][1]:
                intervals[i] = [min(intervals[i][0], intervals[i-1][0]), max(intervals[i][1], intervals[i-1][1])]
                i+=1
            else:
                s.append(intervals[i-1])
                i+=1
        s.append(intervals[i-1])
        return s
print(merge([[1,4],[4,5],[6,6],[3,10]]))