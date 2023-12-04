####动态规划+链表+回溯算法+二分法
#####二分查找
# def erfen(nums,target):
#     i=0
#     j=len(nums)-1
#     while i<=j:
#         mid=(i+j)//2
#         if nums[mid]>target:
#             j=mid-1
#         elif nums[mid]<target:
#             i=mid+1
#         else:
#             return mid
#     return mid

# ####链表的学习
# class Node():
#     def __init__(self,val):
#         self.val=val
#         self.next=None
# s=Node(-101)
# a=Node(1)
# b=Node(2)
# c=Node(3)
# d=Node(4)
# e=Node(4)
# s.next=a
# a.next=b
# b.next=c
# c.next=d
# d.next=e
#
# #
# def dayin(head):
#     while head is not None:
#         print(head.val,end=' ')
#         head=head.next
# dayin(a)
# print()
# print('===================================')
#
# def deleteDuplicates(head):
#     if head==None or head.next==None:
#         return head
#     elif head.next.next==None:
#         if head.next.val==head.val:
#             return None
#         else:
#             return head
#     else:
#         pre = head
#         cai = pre  ##尾部连接
#         tail = head  ###跟踪
#         tmp = head.next  ###前进
#         while tmp != None:
#             if tmp.next!=None:
#                 if tmp.next.val > tmp.val > tail.val:
#                     cai.next = tmp
#                     cai = tmp
#                     tail = tail.next
#                     tmp = tmp.next
#                 else:
#                     tail=tail.next
#                     tmp=tmp.next
#             else:
#                 if tmp.val == tail.val:
#                     tail=tail.next
#                     tmp=tmp.next
#                 else:
#                     cai.next = tmp
#                     cai = tmp
#                     tail = tail.next
#                     tmp = tmp.next
#         cai.next=None
#         return pre.next
# cc=deleteDuplicates(s)
# dayin(cc)

# del b
# dayin(c)
# def reverseBetween(head,left,right):
#     if left==1:
#         pre,cur=None,head
#         j=0
#         while j < right :  ###完成从left到right的反转
#             tmp = cur.next
#             cur.next = pre
#             pre = cur
#             cur = tmp
#             j += 1
#         head.next=cur
#         return pre
#     elif left == right:
#         return head
#     else:  ###反转链表的写法
#         tail = head
#         i = 1
#         while i != left - 1:
#             tail = tail.next
#             i += 1  ###找到第left-1那个的位置
#         cc = tail.next
#         pre, cur = None, tail.next
#         j = 0
#         while j < right - left + 1:  ###完成从left到right的反转
#             tmp = cur.next
#             cur.next = pre
#             pre = cur
#             cur = tmp
#             j += 1
#         tail.next = pre
#         cc.next = cur
#     return head
# cc=reverseBetween(a,1,2)
# dayin(cc)
# def fun(s,x):
#     s1=[]
#     for i in s:
#         if i<x:
#             s1.append(i)
#             s.remove(i)
#         else:
#             continue
#         c=s1+s
#     print(c)
# fun([1,0,2,4,3,5,2],3)
# def partition(head, x):
#     if head is None:
#         return head
#     else:
#         s = []
#         while head:
#             s.append(head.val)
#             head = head.next
#         s1 = []
#         for i in s:
#             if i < x:
#                 s1.append(i)
#                 s.remove(i)
#             else:
#                 continue
#         c = s1 + s
#         print(c)
#         head1 = Node(c[0])
#         tail = head1
#         for i in c[1:]:
#             node = Node(i)
#             tail.next = node
#             tail = node
#         return head1
# partition([1,4,3,2,5,2],3)
# def search(nums, target):  ###经典二分查找
#     if len(nums) == 1:
#         if nums[0] == target:
#             return True
#         else:
#             return False
#     else:
#         i=0
#         j=len(nums)-1
#         while i<=j:
#             mid=(i+j)//2
#             if nums[mid]>target:
#                 j=mid-1
#             elif nums[mid]<target:
#                 i=mid+1
#             else:
#                 return True
#         return False
# print(search([1,0,1,1,1],0))

# def erfen(nums,target):
#     i=0
#     j=len(nums)-1
#     while i<=j:
#         mid=(i+j)//2
#         if nums[mid]>target:
#             j=mid-1
#         elif nums[mid]<target:
#             i=mid+1
#         else:
#             return mid
#     return 'meiyou'
# s='a b'
# print(s[::-1])
# def lengthOfLastWord(s):
#     ss = ''
#     while s[-1] == ' ':
#         s = s[:len(s) - 1]
#     for i in range(len(s)-1,-1,-1):
#         if s[i] != ' ':
#             ss = ss + s[i]
#         elif s[i]==' ':
#             break
#     return len(ss)
#
# print(lengthOfLastWord('sda jaksdj  '))
# def plusOne(digits):
#     if digits[-1] != 9:
#         digits[-1] = digits[-1] + 1
#         return digits
#     elif digits==[9]:
#         return [1,0]
#     else:
#         digits[-1]=0
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
# matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# for i in matrix[1]:
#     matrix.remove(i)
# print(matrix)
# def chuli(num):
#     num.pop()
#     num.pop(0)
#     if len(num)==0:
#         return []
#     else:
#         for i in range(len(num)):
#             num[i].pop()
#             num[i].pop(0)
#         return num
#
#
# def spiralOrder(matrix):
#     if  len(matrix) == 0 or len(matrix[0])==0:
#         return []
#     elif len(matrix) == 1:  ####一行
#         return matrix[0]
#     elif len(matrix[0]) == 1:  ###一列
#         s = []
#         for i in range(len(matrix)):
#             s = s + matrix[i]
#         return s
#     else:
#         n = len(matrix)  # 行
#         m = len(matrix[0])  # 列
#         s1 = []  ##进行矩阵的储存
#         for i in range(m - 1):
#             s1.append(matrix[0][i])
#         for i in range(n - 1):
#             s1.append(matrix[i][m - 1])
#         for i in range(m - 1, 0, -1):
#             s1.append(matrix[n - 1][i])
#         for i in range(n - 1, 0, -1):
#             s1.append(matrix[i][0])
#     s1 = s1 + spiralOrder(chuli(matrix))
#     return s1
# print(spiralOrder([[1,2],[3,4],[5,6],[7,8]]))
# def climbStairs(n):
#     if n == 1:
#         return 1
#     elif n == 2:
#         return 2
#     elif n == 3:
#         return 3
#     else:
#         s={}
#         if ''
#         return climbStairs(n-1)+2*climbStairs(n-2)
# print(climbStairs(4))
# s={}
# s['climbStairs(4)']=climbStairs(4)
# print(s)
@lru_cache(maxsize=None)
def factorial(n):
   if n <1:   # base case
       return 1
   else:
       return n * factorial( n - 1 )  # recursive call




