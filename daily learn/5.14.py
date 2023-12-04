# -*- coding:utf-8 -*-
# @Time      :2022/5/14 10:39
# @Author    :Riemanner
# Write code with comments !!!
# def computeArea(ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
#     s1 = abs(ax1 - ax2) * abs(ay1 - ay2)  ###第一个面积
#     s2 = abs(bx1 - bx2) * abs(by1 - by2)  ###第二个面积
#     if bx2 < ax1 or bx1 > ax2:  ###没有相交
#         return s1 + s2
#     else:
#         s11 = [ax1, ax2, bx1, bx2]
#         s22 = [ay1, ay2, by1, by2]
#         s11.sort()
#         s22.sort()
#         return s1 + s2 - (abs(s11[2] - s11[1]) * abs(s22[2] - s22[1]))
# print(computeArea(-2,-2,2,2,-1,4,1,6))
# def calculate(s):
#     stack = []
#     pre_op = '+'
#     num = 0
#     for i, each in enumerate(s):
#         if each.isdigit():
#             num = 10 * num + int(each)
#         if i == len(s) - 1 or each in '+-*/':
#             if pre_op == '+':
#                 stack.append(num)
#             elif pre_op == '-':
#                 stack.append(-num)
#             elif pre_op == '*':
#                 stack.append(stack.pop() * num)
#             elif pre_op == '/':
#                 top = stack.pop()
#                 if top < 0:
#                     stack.append(int(top / num))
#                 else:
#                     stack.append(top // num)
#             pre_op = each
#             num = 0
#     return sum(stack)
# print(calculate('5415/3251*21'))
# def calculate(s):
#     shuzi_stack = []
#     fuhao_stack= []
#     cun_chu=''
#     for i in range(len(s)):
#         if s[i].isdigit():
#             cun_chu+=s[i]
#             continue
#         elif s[i] in '*/+-()':
#             fuhao_stack.append(s[i])
#             if len(cun_chu)!=0:
#                 shuzi_stack.append(int(cun_chu))
#                 cun_chu=''
#     if len(cun_chu)!=0:
#         shuzi_stack.append(int(cun_chu))
#     print(fuhao_stack)
#     print(shuzi_stack)
# print(calculate("(1+(4+5+2)-3)+(6+8)"))
# def function(s):
#     stack=[]
#     cun_shu=''
#     for i in range(len(s)):
#         if s[i].isdigit():
#             cun_shu+=s[i]
#         elif s[i] in '()+-':
#             if len(cun_shu)!=0:
#                 stack.append(int(cun_shu))
#             cun_shu=''
#     if len(cun_shu)!=0:
#         stack.append(cun_shu)
#     print(stack)
# function('1525+14+2-(12+345+4)')

# def isPowerOfTwo(n):
#     if (n % 2 != 0 and n!=1) or n==0 :
#         return False
#     else:
#         while n != 1 and n != -1:
#             if n % 2 == 0:
#                 n = int(n / 2)
#             else:
#                 return False
#         return True
# print(isPowerOfTwo(-2))
import math
def isPowerOfTwo(n: int) -> bool:
    if n <= 0:
        return False
    else:
        return round(math.log(n, 2),14) % 1 == 0
print(isPowerOfTwo(32767))

print(math.log(32767,2))












