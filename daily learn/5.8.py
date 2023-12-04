# -*- coding:utf-8 -*-
# @Time      :2022/5/8 9:25
# @Author    :Riemanner
# Write code with comments !!!
# def ziji(nums):不重复与重复
#     res=[]
#     def bt(start,path,nums):
#         res.append(path[:])
#         for i in range(start,len(nums)):
#             if i==start or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 bt(i+1,path,nums[:i]+nums[i+1:])
#                 path.pop()
#     bt(0,[],nums)
#     return res
# print(ziji([1,1,3,3,4]))
# def palie(nums):重复与不重复已经解决
#     res=[]
#     n=len(nums)
#     def bt(path,nums):
#         if not nums:
#             res.append(path[:])
#         for i in range(len(nums)):
#             if i==0 or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 bt(path,nums[:i]+nums[i+1:])
#                 path.pop()
#     bt([],nums)
#     return res
# print(palie([1,1,3]))
# def zuhezuida(nums):
#     nums = list(map(str,nums))
#     nums.sort(reverse=True,key=lambda x:int(x+str(x[-1])*(9-len(x))))
#     return ''.join(nums)
# print(zuhezuida(nums = [34323,3432,9]))
# def maopao(nums):
#     nums = list(map(str, nums))
#     for i in range(len(nums)):
#         for j in range(len(nums)-1,i,-1):
#             if int(nums[j]+nums[j-1])>int(nums[j-1]+nums[j]):
#                 nums[j-1],nums[j]=nums[j],nums[j-1]
#     return ''.join(nums)
# print(maopao([10,2,9,39,17]))
# def zuaho(nums):
#     res=nums[0]*nums[1]
#     for i in range(1,len(nums)-1):
#         res=max(res,nums[i]*nums[i+1])
#     return res
# print(zuaho(nums = [-2,0,-1]))
# def maxProduct(A):
#     B = A[::-1]
#     for i in range(1, len(A)):
#
#     return max(max(A), max(B))
# print(maxProduct([2,3,-2,8]))
# def zhihsu(s):
#     if s==1:
#         return False
#     else:
#         for i in range(2,s):
#             if s%i==0:
#                 return False
#         return True
# def countPrimes(n):
#     mome={0:0,1:0,2:1}
#     def dps(n,mome):
#         if n in mome:
#             return mome[n]
#         else:
#             while not zhihsu(n-1):
#                 n-=1
#             mome[n-1]=dps(n-1,mome)
#             print(mome)
#             return mome[n]
#     return dps(n,mome)
# print(countPrimes(4))
# from functools import cache
# @cache
# def aishi(n):
#     k=n-2
#     nums=list(range(2,n))
#     for i in range(len(nums)):
#         if nums[i]==0:
#             continue
#         s = nums[i]
#         while i<len(nums)-1:
#             if nums[i+1]%s==0 and nums[i+1]!=0:
#                 nums[i+1]=0
#                 k-=1
#                 i+=1
#             else:
#                 i+=1
#     return k
# print(aishi(100000))
# from functools import cache
# @cache
# def countPrimes(n: int) -> int:
#     if n <= 2:
#         return 0
#     else:
#         l = [False] * 2 + [True] * (n - 2)
#     k=0
#     for i in range(2, n):
#         if l[i] == True:
#             k+=1
#             for j in range(2 * i, n, i):
#                 l[j] = False
#     return k
# print(countPrimes(6285450))
from functools import cache
# def fanhzauliebiao(nums,k):
#     k=k%len(nums)
#     c=k
#     while k!=0:
#         nums.append(0)
#         k-=1
#     nums[c:]=nums[:len(nums)-c]
#     nums[:c]=nums[len(nums)-c:]
#     while c!=0:
#         nums.pop()
#         c-=1
#     return numsf
from functools import cache
@cache
def jiecheng(n):
    if n==1:
        return 1
    else:
        return (n*jiecheng(n-1))
print(jiecheng(300))
def fanhui(n):
    z5=0
    z10=0
    z25=0
    for i in range(5,n+1,5):
        if i%10==0 and i%25!=0:
            z10+=1
        elif i%25==0:
            z25+=2
        else:
            z5+=1
    return z5+z10+z25
print(fanhui(300))

