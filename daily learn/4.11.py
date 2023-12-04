####今日进行贪心算法的学习+leecode的两个题#
#0-1背包与分数背包，
#分数背包代码  lamuda函数还是比较优异i是
# goods=[(60,10),(100,20),(120,30),(34,8),(23,1),(145,63),(75,6)]
# def fun(goods,w):
#     val=0
#     goods.sort(key=lambda x:x[0]/x[1],reverse=True)
#     for i,(prize,weight) in enumerate(goods):
#         if w>=weight:
#             w=w-weight
#             val=val+prize
#             print('(%d,%d)这个商品拿了%d'%(prize,weight,weight))
#         elif w<weight and w!=0:
#             val=val+w*(prize/weight)
#             print('(%d,%d)这个商品拿了%d'%(prize,weight,w))
#             w=0
#     print('最多可以拿%d价值的商品'%val)
# fun(goods,56)
# x=[[1,2],[3,4]]
# s=lambda x:x[0][0]+x[1][1]
# print(s(x))
# s1='98'
# s2='89'
# print(s1>s2)
# ####map函数的使用
# a=[1,2,3,4]
# def add(x):
#     if x%2==0:
#         return x
#
# b=list(map(add,a))
# print(b)
#lamuda函数的使用
# x=[1,2,3]
# s=lambda x:x[0]+x[1]-x[2]
# print(s(x))
#要善于使用map和lamuda来进行代码的实现，一方面确实简洁
# s=[(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)]
###在0-16分钟里最多可以举办多少场活动
# def pai(s):
#     if len(s)==1:
#         return s
#     else:
#         s1=[]
#         s.sort(key=lambda x:x[1])
#         s1.append(s[0])
#         s=[i for i in s if i[0]-s1[-1][1]>0]
#         s1.append(pai(s))
#     return s1
# sss=pai(s)
#
# print(sss[1][1][1][0])
# s=[(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)]
# s.sort(key=lambda x:x[1])
#
# def buduigui(s1):
#     res=[s1[0]]
#     for i in range(1,len(s1)):
#         if s1[i][0]>res[-1][1]:
#             res.append(s1[i])
#     return res
# print(buduigui(s))
# def removeDuplicates(nums) -> int:
#     i = 0
#     j = 1
#     while j < len(nums):
#         if nums[i] != nums[j]:
#             nums[i + 1] = nums[j]
#             i += 1
#             j += 1
#         else:
#             j += 1
#     print(nums)
#     print(i+1)
#
# # removeDuplicates([1,1,2,3])
# def strStr(haystack,needle):
#     if len(needle) == 0:
#         return 0
#     else:
#         i = 0  ###haystack的指针
#         j = 0  ###needle的指针
#         while i < len(haystack):
#             if haystack[i] != needle[j]:
#                 i += 1
#             elif haystack[i] == needle[j]:
#                 if haystack[i :i + len(needle)] == needle:
#                     return i
#                 else:
#                     i += 1
#         return -1
# print(strStr('heoll','osdaasd'))
#
# def jump(nums):
#     if len(nums) == 1:
#         return 0
#     elif len(nums) == 2:
#         return 1
#     else:
#         if nums[0]+1 >= len(nums):
#             return 1
#         i = 0
#         k = 0
#         S = []
#         for j in range(i + 1, i + nums[i]+1):
#             s = j - i + nums[j]
#             S.append((j,s))
#         S.sort(key=lambda x: x[1], reverse=True)
#         k += 1
#         if S[0][0] >= len(nums):
#             return k
#         else:
#             k = k + jump(nums[S[0][0]:])
#             return k
# print(jump([1,1,1,1]))
def combinationSum(candidates,target):
    if len(candidates) == 1:
        if target % candidates[0] == 0:
            return [[candidates[0]] * (target / candidates[0])]
        else:
            return []
    else:
        if candidates[0] > target:
            return []
        else:
            S = []
            for i in range(len(candidates) - 1, -1, -1):
                if target % candidates[i] == 0:
                    S.append([candidates[i]] * (target // candidates[i]))
                elif target % candidates[i] != 0 and (target - candidates[i]) < candidates[0]:
                    continue
                elif target % candidates[i] != 0 and (target - candidates[i]) >= candidates[0]:
                    C = [candidates[i]]
                    C.append(combinationSum(candidates[0:len(candidates)], target - candidates[i]))
                    S.append(i)
        return S

print(combinationSum([2,3,6,7],7))
