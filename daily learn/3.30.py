#今天渐渐的正式开始力扣的学习
# list=eval(input())
# target=int(input())
# def twoSum(list, target):
#     n = len(list)
#     for i in range( n - 1):
#         for j in range(i + 1, n):
#             if list[i] + list[j] == target:
#                 return [i,j]
# # twoSum(list,target)
# def addTwoNumbers(l1,l2):
#     s = []
#     if len(l1) == len(l2):
#         for i in range(len(l1) - 1, -1, -1):
#             if l1[i] + l2[i] < 10:
#                 s.append(l1[i] + l2[i])
#             if l1[i] + l2[i] > 10:
#
#             return s
#     if len(l1) > len(l2):
#         for i in range(len(l2) - len(l1) - 1):
#             l2.insert(i, 0)
#         addTwoNumbers(l1, l2)
# def fun(l1,l2):
#     sum1 = 0
#     sum2 = 0
#     s = []
#     for i in range(len(l1)):
#         sum1 = sum1 + l1[i] * pow(10, i)
#     for j in range(len(l2)):
#         sum2 = sum2 + l2[j] * pow(10, j)
#     sum3 = str(sum1 + sum2)
#     for k in range(len(sum3) - 1, -1):
#         s.append(sum3[k])
#     return s
# def fun(nums1,nums2):
#     s = [0] * 10
#     for i in nums2:
#         nums1.append(i)
#     for i in nums1:
#         s[i] += 1
#     nums1=[]
#     for inx, val in enumerate(s):
#         for i in range(val):
#             nums1.append(inx)
#     n = len(nums1)
#     if n % 2 == 0:
#         return (nums1[int(n/2)]+nums1[int((n/2) - 1)])/2
#     if n % 2 != 0:
#         return (nums1[int((n-1)/2)])
# l1=[1,3,5]
# l2=[2,4]
# fun(l1,l2)
# def fun(nums1,nums2):
#         s=[]
#         x=len(nums1)
#         y=len(nums2)
#         i=0
#         j=0
#         while i <x and j<y:
#             if nums1[i]<nums2[j]:
#                 s.append(nums1[i])
#                 i+=1
#             else:
#                 s.append(nums2[j])
#                 j+=1
#         if i<x:
#             for i in range(i,x):
#                 s.append(nums1[i])
#         if j<y:
#             for j in range(j,y):
#                 s.append(nums2[j])
#         print(s)
#         n=len(s)
#         if n % 2 == 0:
#             print((s[int(n/2)]+s[int((n/2) - 1)])/2)
#         if n % 2 != 0:
#             print(s[int((n-1)/2)])
#
# s1=[1,2]
# s2=[3,4]
# fun(s1,s2)
s = {1: 'I', 5: 'V', 10: 'X', 50: 'L', 100: 'C', 500: 'D', 1000: 'M', 4: 'IV', 9: 'IX', 40: 'XL', 90: 'XC', 400: 'CD',
     900: 'CM'}


def fun1(num):  # 定义一位数
    if num == 9 or num == 4 or num == 5:
        return s[num]
    elif num < 5:
        return 'I' * num
    elif num > 5:
        return 'v' + 'I' * (num - 5)


def fun2(num):  # 定义两位数
    if num == 40 or num == 50 or num == 90:
        return s[num]
    elif num < 50:
        return 'X' * (num // 10) + fun1(num - (num // 10) * 10)
    elif num > 50:
        return 'L' + 'X' * ((num // 10) - 5) + fun1(num - (num // 10) * 10)


def fun3(num):  # 定义三位数
    if num == 400 or num == 500 or num == 900 or num == 100:
        return s[num]
    elif num < 500:
        return 'C' * (num // 100) + fun2(num - (num // 100) * 100)
    elif num > 500:
        return 'D' + 'C' * ((num // 100) - 5) + fun2(num - (num // 100) * 100)


def fun4(num):  # 定义四位数
    if num == 1000:
        return s[1000]
    elif num > 1000:
        return 'M' * (num // 1000) + fun3(num - (num // 1000) * 1000)


def intToRoman(self, num: int) -> str:
    if num > 0 and num < 10:
        print(fun1(num))
    if num > 10 and num < 100:
        print(fun2(num))
    if num > 100 and num < 1000:
        print(fun3(num))
    if num > 1000:
        print(fun4(num))
