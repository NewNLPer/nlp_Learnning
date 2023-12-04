####寻找缺失的第一个正数
# def firstMissingPositive(nums):
#     nums.sort()
#     for i in nums:
#         if i <= 0 or i >= len(nums):
#             continue
#         else:
#             nums[i-1] = i
#     for i in range(1, len(nums)+1):
#         if nums[i-1] != i:
#             return i
#     else:
#         return len(nums)+1
# print(firstMissingPositive([-1,-2,5,6,8,1,0,2,3]))
###针对回溯算法再进行体会
# def kuohaoyouxiao(st):
#     if len(st)==0:
#         return
#     elif len(st)==1 or st[0]==')' or len(st)%2!=0:
#         return False
#     else:
#         s=[]
#         for i in st:
#             if i=='(':
#                 s.append(i)
#             else:
#                 if len(s)!=0:
#                     s.pop()
#                 else:
#                     return False
#     if len(s)==0:
#         return True
#     else:
#         return False
#
#
# def longestValidParentheses(s):
#     if len(s) == 0:
#         return 0
#     elif len(s) == 1:
#         return 0
#     else:
#         ss = 0
#         for i in range(len(s)):
#             if s[i] == ')':
#                 continue
#             for j in range(len(s)-1, i,-1):
#                 if s[j] == '(':
#                     continue
#                 elif kuohaoyouxiao(s[i:j + 1]) == False:
#                     continue
#                 elif kuohaoyouxiao(s[i:j + 1]) == True:
#                     if j-i+1>ss:
#                         ss=(j - i + 1)
#         return ss
# print(longestValidParentheses(")(())))(())())"))
#
# def zichan(nums):
#     i=0
#     k=0
#     while i<len(nums):
#         if nums[i]==1:
#             i+=1
#         else:
#             j=i+1
#             for p in range(j,len(nums)):
#                 if nums[p]==0:
#                     continue
#                 if nums[p]!=0:
#                     if p-j+1>k:
#                         k=p-j+1
#                     i=p
#                     break
#             else:
#                 if len(nums)-j+1>k:
#                     k=len(nums)-j+1
#                     return k
#                 else:
#                     return k
#     return k
# def kuohaoyouxiao(st):###匹配括号变0的问题
#     s=[]
#     for i in st:
#         if i=='(':
#             s.append(i)
#         else:
#             if len(s)!=0:
#                 s.pop()
#             else:
#                 return False
#     if len(s)==0:
#         return True
#     else:
#         return False
# def longestValidParentheses(s):
#     res = 0
#     stack = [-1]
#     for i, c in enumerate(s):
#         if c == "(":
#             stack.append(i)
#         else:
#             stack.pop()
#             if not stack:
#                 stack.append(i)
#             else:
#                 res = max(res, i - stack[-1])
#     return res
# print(longestValidParentheses(')))))'))
def kuohao(s):
    ss=[s[0]]
    res=0
    for i in range(1,len(s)):
        if len(ss)==0:
            ss.append(s[i])
            continue
        if ss[-1]+s[i]=='()':
            ss.pop()
            if i-len(ss)+1>res:
                res=i-len(ss)+1
        else:
            ss.append(s[i])
    return res
print(kuohao('()(())'))
