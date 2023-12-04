#AVL树的学习，对于链表，队列，栈，堆，二叉树，黑红树，avl树
#贪心算法类似于小案例，但是还是要理解明白
# t=[100,50,20,5,1]
# def zhaoqian(t,n):
#     m=[0,0,0,0,0]
#     for i,money in enumerate(t):
#         m[i]=n//money
#         n=n%money
#     print(m)
# zhaoqian(t,3212)
# def isMatch(s,p):
#     if p == '.*':
#         return True
#     elif p.isalpha() == True:
#         if s == p:
#             return True
#         else:
#             return False
#     else:
#         i = len(p) - 1
#         j = len(s) - 1
#         while p[i] == s[j] and i >= 0 and j >= 0:
#             while p[i] == s[j] or p[i] == '.':
#                 i -= 1
#                 j -= 1
#             if p[i] != '.' and p[i] != '*' and p[i] != s[j]:
#                 return False
#             elif p[i] == '*':
#                 ss = ''
#                 i -= 1
#                 if s[j] == p[i]:  ##匹配多个
#                     while s[j] == p[i]:
#                         ss = ss + s[j]
#                         j-=1
#                         if j<0:
#                             j=0
#                     i -= 1
#                     if len(ss)==1:
#                         return False
#                     else:
#                         pass
#                 elif s[j] != p[i]:
#                     i -= 1
#                     if p[i] == '*':
#                         i -= 1
#                 if s[j] != p[i]:
#                     return False
#         return True
# print(isMatch('aab','c*a*b'))
# def fun1(s,a):
#     if a.count(s[0])==len(a) and (len(a)>2 or len(a)==0):
#         return True
#     else:
#         return False
# print(fun1('3*',''))
# ##定义n各***
# def funn(s,a):
#     n=len(s)
#     for i in range(1,n,2):
#         for j in range(0,n,2):
#
#
#         if a.count(s[i-1])==len(a) and (len(a)>2 or len(a)==0):

# def isMatch(s,p):
#     if p == '.*':
#         return True
#     elif p.isalpha() == True:
#         if s == p:
#             return True
#         else:
#             return False
#     else:
#         i = len(p) - 1
#         j = len(s) - 1
#         ss = ''
#         while j < 0:
#             while (p[i] == s[j] or p[i]=='.') and j > 0:
#                 ss = ss + s[j]
#                 i -= 1
#                 j -= 1
#             if p[i] == '*':
#                 i -= 1
#             while s[j] == p[i]:
#                 ss = ss + s[j]
#                 j -= 1
#             else:
#                 i -= 1
#                 if p[i] == '*':
#                     i -= 1
#                 else:
#                     return False
#     if ss[::-1]==s:
#         return True
# print(isMatch('a*','aa'))
# def isMatch(s,p):
#     if p == '.*':
#         return True
#     elif p.isalpha() == True:
#         if s == p:
#             return True
#         else:
#             return False
#     elif len(s)==0:
#         if p.count('*')==len(p)/2:
#             return True
#         else:
#             return False
#     else:  # 双指针调走
#         i = len(p) - 1
#         j = len(s) - 1
#         while i > 0 or j > 0:
#             while s[j] == p[i] or p[i]=='.':
#                 j -= 1
#                 i -= 1
#             if p[i] == '*':
#                 i -= 1
#             if s[j] == p[i]:  # 匹配多个
#                 if s[j - 1] != p[i]:
#                     return False
#                 while s[j] == p[i] and j < 0:
#                     j -= 1
#                 i -= 1
#             if s[j] != p[i] and i>0:  # 匹配0个
#                 i -= 1
#             else:
#                 return False
#         return True
# print(isMatch('aaaaaaa','a*'))
# def fun1(s,a):
#     if a.count(s[0])==len(a) and (len(a)>2 or len(a)==0):
#         return True
#     else:
#         return False
# print(fun1('3*',''))
# ##定义n各***
# def fun1(s,a):##生成有效字母
#     n=len(s)
#     ss=''
#     for i in range(0,n,2):
#         if s[i] in a:
#             ss=ss+s[i]+'*'
#     return ss
# def fun2(s,a):##根据有效字母进行判断
#     n=len(s)
#     S=[[]]*(len(s)/2)
#     for i in range(1,n,2):
#         for j in range(1,len(a)):
#             if a[i]==s[i] and a[i-1]==s[i]:
# def isMatch(s,p):
#     if p == '.*':
#         return True
#     elif p.isalpha() == True:
#         if s == p:
#             return True
#         else:
#             return False
#     else:  # 双指针调走
#         i = len(p) - 1
#         j = len(s) - 1
#         while i!=0 and j!=0:
#             while s[j] == p[i] or p[i]=='.':
#                 j -= 1
#                 i -= 1
#             if p[i] == '*':
#                 i -= 1
#             if s[j] == p[i]:  # 匹配多个
#                 if s[j - 1] != p[i]:
#                     return False
#                 while s[j] == p[i] and j > 0:
#                     j -= 1
#                 if i > 0:
#                     i -= 1
#             if s[j] != p[i] and i > 0:  # 匹配0个
#                 i -= 1
#             if i+j==0:
#                 return True
#         return False
#
# print(isMatch('aaaBB','a*B*'))ds=[]
# def longestCommonPrefix(strs):
#     s=[]
#     k = 0
#     n = len(min(strs, key=len))
#     for i in range(n):
#         s = []
#         for j in range(len(strs)):
#             s.append(strs[j][i])
#         if len(set(s)) == 1:
#             k += 1
#         if len(set(s)) != 1:
#             break
#     return strs[0][0:k]
# print(longestCommonPrefix(['fsead','fsea']))
# def threeSum(nums):
#     S = []
#     if len(nums) < 3 or max(nums) < 0 or min(nums) > 0:
#         return []
#     elif min(nums) == max(nums) == 0 and len(nums) >= 3:
#         return [[0, 0, 0]]
#     else:
#         k = 0
#         l = 1
#         S = []
#         while k < len(nums) or l < len(nums):
#             for j in range(l + 1,: ):
#             if -(nums[k] + nums[l]) in nums[l + 1:]:
#                 if sorted([num[k], nums[l], -(nums[k] + nums[l]) not in S
#                     S.append(sorted([nums[k], nums[l], -(nums[k] + nums[l])]))
#                 k += 1
#                 l += 1
#                 break