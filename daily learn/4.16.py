# ###今天对项目的代码进行拼接，然后对最近做的leecode进行复习
# def rotate(matrix):
#     """
#     Do not return anything, modify matrix in-place instead.
#     """
#     if len(matrix) == 1:
#         return matrix
#     else:
#         for i in range(len(matrix) // 2):
#             matrix[i], matrix[len(matrix) - i - 1] = matrix[len(matrix) - i - 1], matrix[i]
#         for i in range(len(matrix)):
#             for j in range(i,len(matrix)):
#                 matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
#     return matrix
# print(rotate([[1,2,3],[4,5,6],[7,8,9]]))

def function(st):##双指针解法
    st=str(st)
    i=0
    j=1
    s=[]
    while i<len(st) and j<len(st):
        if st[i]==st[j]:
            j+=1
        elif st[i]!=st[j]:
            s.append(str(j-i))
            s.append(str(st[i]))
            i=j
            j=i+1
    s.append(str(j-i))
    s.append(st[i])
    return ''.join(s)
def countAndSay(n):###递归
    if n==2:
        return 11
    else:
        return function(countAndSay(n-1))


print(countAndSay(3))
