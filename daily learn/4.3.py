#队列、栈（括号匹配+迷宫问题）、链表
# def fun(num):
#     if num<0:
#         return -1
#     if num>0:
#         return 1
# def divide(dividend,divisor):
#     s=0
#     ss=2
#     if divisor==1:
#         if dividend>2147483647:
#             return 2147483647
#         elif dividend<-2147483648:
#             return -2147483648
#         else:
#             return dividend
#     elif divisor==-1:
#         if dividend>2147483647:
#             return -2147483648
#         elif dividend<=-2147483647:
#             return 2147483647
#         else:
#             return -dividend
#     elif abs(dividend)<abs(divisor):
#         return 0
#     else:
#         s1=abs(dividend)
#         s2=abs(divisor)
#         s3=abs(divisor)
#         if s2+s2>s1:
#             return 1*fun(divisor)*fun(dividend)
#         else:
#             while s2+s2<=s1:
#                 s2=s2+s2
#                 s+=1
#             for i in range(s-1):
#                 ss=ss+ss
#             ss=ss+divide(s1-s2,s3)
#             ss=fun(divisor)*fun(dividend)*ss
#             if ss>2147483647:
#                 return 2147483647
#             elif ss<-2147483648:
#                 return -2147483648
#             else:
#                 return ss
# print(divide(10,-10))
# def rotate(matrix) -> None:
#     """
#     Do not return anything, modify matrix in-place instead.
#     """
#     s = matrix
#     for i in range(len(s)):
#         for j in range(len(s)):
#             matrix[i][j] = s[len(s) - 1 - j][i]
#     print(matrix)
# rotate([[1,2,3],[4,5,6],[7,8,9]])
