# -*- coding:utf-8 -*-
# @Time      :2022/4/24 11:20
# @Author    :Riemanner
# Write code with comments !!!
####关于迷宫与螺旋矩阵
####备忘录+动态规划
# def plusOne(digits):
#     if digits[-1] != 9:
#         digits[-1] = digits[-1] + 1
#         return digits
#     else:
#         digits[-1] = 0
#         for i in range(len(digits) - 2, -1, -1):
#             if digits[i] == 9:
#                 digits[i] = 0
#                 if i == 0:
#                     digits.insert(0, 1)
#                 continue
#             else:
#                 digits[i] = digits[i] + 1
#                 break
#         return digits
# print(plusOne([9,9]))
# maze=[
#     [1,1,1,1,1,1,1,1,1,1],
#     [1,0,0,1,0,0,0,1,0,1],
#     [1,0,0,1,0,0,0,1,0,1],
#     [1,0,0,0,0,1,1,0,0,1],
#     [1,0,1,1,1,0,1,0,0,1],
#     [1,0,0,0,1,0,0,0,0,1],
#     [1,0,1,0,0,0,1,0,0,1],
#     [1,0,1,0,1,1,1,1,0,1],
#     [1,1,0,0,0,1,0,0,0,1],
#     [1,1,1,1,1,1,1,1,1,1]
# ]
# dirs=[
#     lambda x,y:(x,y+1),
#     lambda x,y:(x+1,y),
#     lambda x,y:(x,y-1),
#     lambda x,y:(x-1,y)
# ]
# def maze_path(x1,y1,x2,y2):#起点终点坐标
#     stack=[]
#     stack.append((x1,y1))
#     while len(stack)>0:
#        curNode=stack[-1]
#        if curNode[0]==x2 and curNode[1]==y2:###通过while不断进行迭代，对是否到了终点进行判断
#            for p in stack:
#                p=str(p)
#                print('%s---->'%p,end=' ')
#            return True
#        for dir in dirs:
#            nextNode=dir(curNode[0],curNode[1])
#            if maze[nextNode[0]][nextNode[1]]==0:
#                stack.append(nextNode)
#                maze[nextNode[0]][nextNode[1]]=2#表示已经走过：
#                break
#        else:
#            # maze[nextNode[0]][nextNode[1]]=2
#            stack.pop()
#     else:
#         print('无路可走')
#         return False
# maze_path(1,1,8,8)
# dirs=[
#     lambda x,y:(x,y+1),
#     lambda x,y:(x+1,y),
#     lambda x,y:(x,y-1),
#     lambda x,y:(x-1,y)
# def maze_path(n):#起点终点坐标
#     s = list(range(1, n * n + 1))
#     ss=[[0,0,0],[0,0,0],[0,0,0]]
#     i=0
#     j=0
#     while len(s)>0:
#        for dir in dirs:
#            nextNode=dir(i,j)
#            if maze[nextNode[0]][nextNode[1]]==0:
#                stack.append(nextNode)
#                maze[nextNode[0]][nextNode[1]]=2#表示已经走过：
#                break
#        else:
#            # maze[nextNode[0]][nextNode[1]]=2
#            stack.pop()
#     else:
#         print('无路可走')
#         return False
# def generateMatrix(n):
#     s = list(range(1, n * n + 1))
#     ss = [[0,0,0],[0,0,0],[0,0,0]]
#     i = 0
#     j = 0
#     while len(ss) > 0:
#         if j < n and ss[i][j] == 0:
#             ss[i][j] = s[0]
#             if j <n-1:
#                 j += 1
#             else:
#                 j=j
#             s.pop(0)
#         elif i < n and ss[i+1][j] == 0:
#             i += 1
#             ss[i][j] = s[0]
#             s.pop(0)
#         elif i >= 0 and s[i][j-1] == 0:
#             j-=1
#             ss[i][j] = s[0]
#             s.pop(0)
#         elif j >= 1 and s[i][j] == 0:
#             ss[i][j] = s[0]
#             j -= 1
#             s.pop(0)
#     return ss
# print(generateMatrix(3))
# def generateMatrix(n):
#     s = list(range(1, n * n + 1))
#     ss = []
#     for i in range(n):
#         ss.append([0]*n)
#     i = 0
#     j = 0
#     ss[0][0] = 1
#     s.pop(0)
#     while len(s) > 0:
#         while j < n - 1 and ss[i][j + 1] == 0:
#             ss[i][j + 1] = s[0]
#             s.pop(0)
#             j += 1
#         while i < n - 1 and ss[i + 1][j] == 0:
#             ss[i + 1][j] = s[0]
#             s.pop(0)
#             i += 1
#         while j > 0 and ss[i][j - 1] == 0:
#             ss[i][j - 1] = s[0]
#             s.pop(0)
#             j -= 1
#         while i > 0 and ss[i - 1][j] == 0:
#             ss[i-1][j]=s[0]
#             s.pop(0)
#             i -= 1
#     return ss
# print(generateMatrix(4))
import time

# 2.带备忘录的斐波那契数列， 时间复杂度为 O(n)
# class Solution2:
#     def fib(self, n):
#         help_dict = {1: 1, 2: 2}
#         return self.helper(n, help_dict)   #####标准递归函数
#     def helper(self, n, help_dict):
#         if n == 1 or n == 2:
#             return 1
#         if n in help_dict:
#             return help_dict[n]
#         else:
#             help_dict[n] = self.helper(n - 1, help_dict) + self.helper(n - 2, help_dict)#####递归关系
#             # print(help_dict)
#             return self.helper(n - 1, help_dict) + self.helper(n - 2, help_dict)####返回结果
#
# print(Solution2().fib(10))
#######链表反转
# class Node():
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
# ####显示
# def dayin(head):
#     while head :
#         print(head.val,end=' ')
#         head=head.next
# dayin(a)
# print()
# def fnazhuan(head):
#     pre=None
#     cur=head
#     while cur:
#         tmp=cur.next
#         cur.next=pre
#         pre=cur
#         cur=tmp
#     return pre
# cc=fnazhuan(a)
# dayin(cc)

######hash table 存储
####利用 hash table 的话，避免重复计算，大致存在两个函数，表示递归与存储量
# class Solutions:
#     def fac(self,n):
#         hash_table={1:1,2:1}
#         return self.help(n,hash_table)
#     def help(self,n,hash_table):
#         if n==1 or n==2:
#             return 1
#         elif n in hash_table:
#             return hash_table[n]
#         else:
#             hash_table[n]=self.help(n-1,hash_table)+self.help(n-2,hash_table)
#             return self.help(n-1,hash_table)+self.help(n-2,hash_table)
# print(Solutions().fac(100))
# def insert(intervals,newInterval):
#     if len(intervals)==0:
#         return [newInterval]
#     elif len(intervals)==1:
#         if newInterval[0]>intervals[0][1] or newInterval[1]<intervals[0][0]:
#             s=[intervals[0],newInterval]
#             s.sort(key=lambda x:x[0])
#             return s
#         else:
#             return [[min(newInterval[0],intervals[0][0]),max(newInterval[1],intervals[0][1])]]
#     else:
#         if newInterval[1]<intervals[0][0]:
#             return intervals.insert(newInterval,0)
#         else:
#         s = []
#         for i in range(len(intervals)):
#             if intervals[i][1] < newInterval[0]:
#                 s.append(intervals[i])
#             else:
#                 s1 = min(intervals[i][0], newInterval[0])  #####确定区间左断点,然后再确定另一个端点
#                 c = len(intervals) - 1
#                 while c >=i:
#                     if intervals[c][0] > newInterval[1]:
#                         s.append(intervals[c])
#                         c -= 1
#                     else:
#                         s2 = max(newInterval[1], intervals[c][1])#####，从最右边开始，确定区间右断点,然后再确定
#                         s.append([s1, s2])
#                         break
#                 break
#
#         else:
#             s.append(newInterval)
#     s.sort(key=lambda x: x[0])
#     return s
# print(insert([[1,5]],[0,0]))
# def merge(self, intervals: List[List[int]]) -> List[List[int]]:
#     if len(intervals) == 1:
#         return intervals
#     else:
#         s=[]
#         i=1
#         intervals.sort(key=lambda x: x[0])
#         while i<len(intervals):
#             if intervals[i][0]<= intervals[i-1][1]:
#                 intervals[i] = [min(intervals[i][0], intervals[i-1][0]), max(intervals[i][1], intervals[i-1][1])]
#                 i+=1
#             else:
#                 s.append(intervals[i-1])
#                 i+=1
#         s.append(intervals[i-1])
#         return s
# class Solution:
#     def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
#         return merge(intervals+[newInterval])

# def merge(intervals):
#     if len(intervals) == 1:
#         return intervals
#     else:
#         s = []
#         i = 1
#         intervals.sort(key=lambda x: x[0])  ##先进行排序
#         while i < len(intervals):
#             if intervals[i][0] <= intervals[i - 1][1]:
#                 intervals[i] = [min(intervals[i][0], intervals[i - 1][0]), max(intervals[i][1], intervals[i - 1][1])]
#                 i += 1
#             else:
#                 s.append(intervals[i - 1])
#                 i += 1
#         s.append(intervals[i - 1])
#         return s
#
#
# print(merge(intervals = [[1,3],[2,6],[8,10],[15,18]]))
import copy
# def combine(n,k):
#     def bt(n, path, start):
#         if len(path) == n:  # 组合中的元素个数为k时，加入结果集并剪枝
#             res.append(copy.deepcopy(path))
#             return
#         for i in range(start, n + 1):  # 遍历选择列表
#             path.append(i)  # 做出选择
#             bt(n, path, i + 1)  # 进入下一层，注意i+1
#             path.pop()  # 撤销选择
#
#     res = []  # 结果集
#     bt(n, [], 1)  # 开始递归\
#     print(res)
# print(combine(3,2))
# class Solution:
#     def permute(self, nums,k):
#
#         def bt(nums, path):
#             if len(path) == len(nums): # 所有数排列完毕，加入结果集
#                 res.append(copy.deepcopy(path))
#             for i in range(len(nums)): # 遍历选择列表
#                 if nums[i] not in path: # path中存在的数不能被列入选择列表
#                     path.append(nums[i]) # 做出选择
#                     bt(nums, path) # 进入下一层
#                     path.pop() # 撤销选择
#                 if i>k:
#                     break
#         res = [] # 结果集
#         bt(nums, []) # 开始递归
#         # return "".join(map(str, res[k-1])) # 返回结果集
#         print(res)
# print(Solution().permute(nums=[2,1,3],k=3))
# def setZeroes(matrix):
#     m = len(matrix)  # 行
#     n = len(matrix[0])
#     # 列
#     s1 = set()  # 记录行
#     s2 = set()  # 记录列
#     for i in range(m):
#         for j in range(n):
#             if matrix[i][j] == 0:
#                 s1.add(i)
#                 s2.add(j)
#     for i in range(m):
#         if i in s1:
#             matrix[i] = [0] * n
#         for j in s2:
#             matrix[i][j] = 0
#     return matrix
# print(setZeroes(matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]))
# def canConstruct(ransomNote,magazine) :
#     s1 = {}
#     s2 = {}
#     for i in range(len(ransomNote)):
#         if ransomNote[i] not in s1:
#             s1[ransomNote[i]] = 1
#         else:
#             s1[ransomNote[i]] += 1
#     for j in range(len(magazine)):
#         if magazine[j] not in s2:
#             s2[magazine[j]] = 1
#         else:
#             s2[magazine[j]] += 1
#     for key in s1:
#         if key not in s2:
#             return False
#         else:
#             if s2[key] < s1[key]:
#                 return False
#     return True
# print(canConstruct('aa','aab'))

class Node():
    def __init__(self,val):
        self.val=val
        self.next=None
a=Node(1)
b=Node(2)
c=Node(3)
d=Node(4)
a.next=b
b.next=c
c.next=d
####显示
def dayin(head):
    while head :
        print(head.val,end=' ')
        head=head.next
dayin(a)
print()
def fnazhuan(head):
    head1=head
    pre=None
    cur=head1
    while cur:
        tmp=cur.next
        cur.next=pre
        pre=cur
        cur=tmp
    return pre
cc=fnazhuan(a)
dayin(cc)




