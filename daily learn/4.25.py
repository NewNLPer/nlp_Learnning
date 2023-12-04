# -*- coding:utf-8 -*-
# @Time      :2022/4/25 11:26
# @Author    :Riemanner
# Write code with comments !!!
#####带备忘录的递归==动态规划
# def fib(n):
#     if n == 1 or n == 2:
#         return 1
#     result = {1: 1, 2: 1}
#     for i in range(3, n + 1):
#         result[i] = result[i - 1] + result[i - 2]
#     print(result)
#     return result[n]
# fib(10)
# def isInterleave(s1,s2, s3):
#     for i in s1:
#         if i in s1:
#             s3 = s3.replace(i, '')
#     if s3 == s2:
#         return True
#     else:
#         return False
# print(isInterleave(s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"))
# def isInterleave(s1, s2, s3):
# ###### 采用三指针分别对三个字符串进行遍历比较
#     if len(s1)+len(s2)!=len(s3):
#         return False
#     elif (s1=='' and s2!=s3 )or (s2=='' and s1!=s3):
#         return False
#     else:
#         tmp_s1=0
#         tmp_s2=0
#         tmp_s3=0
#         s4=[]
#     while tmp_s3<len(s3):
#         if  tmp_s1<len(s1) and tmp_s2<len(s2) and s1[tmp_s1]==s2[tmp_s2]==s3[tmp_s3]:
#             s4.append(s1[tmp_s1])
#             tmp_s1+=1
#             tmp_s2+=1
#             tmp_s3+=1
#         elif tmp_s1<len(s1) and tmp_s2<len(s2) and  s1[tmp_s1]!=s3[tmp_s3] and s2[tmp_s2]!=s3[tmp_s3] and len(s4)==0:
#             return False
#         else:
#             if tmp_s1<len(s1) and s1[tmp_s1]==s3[tmp_s3]:
#                 tmp_s1+=1
#                 tmp_s3+=1
#             elif tmp_s2<len(s2) and s2[tmp_s2]==s3[tmp_s3]:
#                 tmp_s2+=1
#                 tmp_s3+=1
#             else:
#                 k=0
#                 while tmp_s3<len(s3):
#                     if len(s4)!=0 and s3[tmp_s3]==s4[0]:
#                         tmp_s3+=1
#                         s4.pop(0)
#                         k+=1
#                     elif k==0 and s3[tmp_s3]!=s4[0]:
#                         return False
#                     else:
#                         break
#     if tmp_s3==len(s3) and tmp_s2==len(s2) and tmp_s1==len(s1):
#         return True
#     else:
#         return False
# print(isInterleave("aabd","abdc","aabdbadc"))

# class  Node():
#     def __init__(self,val):
#         self.val=val
#         self.next=None
# a=Node(1)
# b=Node(2)
# c=Node(3)
# d=Node(4)
# a.next=b
# b.next=c
# c.next=d
#
# def dayin(head):
#     while head:
#         print(head.val,end=' ')
#         head=head.next
# dayin(a)
# def fanzhuan(head):
#     pre=None
#     cur=head
#     while cur:
#         tmp=cur.next
#         cur.next=pre
#         pre=cur
#         cur=tmp
#     return pre
# cc=fanzhuan(a)
# print()
# dayin(cc)
####开始进行回溯算法
import copy


def permute(nums):
    def bt(nums, path):
        res.append(copy.deepcopy(path))
        for i in range(len(nums)):  # 遍历选择列表
            if nums[i] not in path:  # path中存在的数不能被列入选择列表
                path.append(nums[i])  # 做出选择
                bt(nums, path)  # 进入下一层
                path.pop()  # 撤销选择

    res = []  # 结果集
    bt(nums, [])  # 开始递归
    return res  # 返回结果集
print(permute([1,2,3]))