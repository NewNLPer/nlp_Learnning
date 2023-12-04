# -*- coding:utf-8 -*-
# @Time      :2022/5/11 11:11
# @Author    :Riemanner
# Write code with comments !!!
# def nthUglyNumber(n):
#     s = [1]
#     one = 0
#     two = 0
#     three = 0
#     deidai =1
#     while deidai <= n-1:
#         if min(s[one] * 2, s[two] * 3, s[three] * 5) == s[one] * 2 :
#             if s[one]*2 !=s[-1]:
#                 s.append(s[one] * 2)
#                 one += 1
#                 deidai += 1
#             else:
#                 one += 1
#         elif min(s[one] * 2, s[two] * 3, s[three] * 5) == s[two] * 3 :
#             if s[two] * 3 != s[-1]:
#                 s.append(s[two] * 3)
#                 two += 1
#                 deidai += 1
#             else:
#                 two += 1
#         elif min(s[one] * 2, s[two] * 3, s[three] * 5) == s[three] * 5:
#             if s[three]*5 != s[-1]:
#                 s.append(s[three] * 5)
#                 three += 1
#                 deidai += 1
#             else:
#                 three += 1
#     return s[-1]
# print(nthUglyNumber(100))
# def jiecheng(n):
#     k=0
#     for i in range(5,n+1,5):
#         while i%5==0:
#             k+=1
#             i=i//5
#     return k
# print(jiecheng(10000))
class Node():
    def __init__(self,val):
        self.val=val
        self.next=None
a=Node(1)
b=Node(2)
c=Node(3)
# d=Node(4)
a.next=b
b.next=c
# c.next=d

A=Node(5)
B=Node(4)
C=Node(8)
# D=Node(40)
A.next=B
# B.next=C
#
#
# ###已给两个链表进行交叉合并处理，
# def jiachahebing(head1,head2):
#     tail = head1
#     while head2:
#         nxt = tail.next
#         tail.next = head2
#         cnext = head2.next
#         head2.next = nxt
#         head2 = cnext
#         tail = nxt
#     return head1
#
def dayin(head):
    while head:
        print(head.val,end=' ')
        head=head.next
# dayin(a)
# print()
# dayin(A)
# print()
# c=jiachahebing(a,A)
# dayin(c)



def merge(head1, head2):
    node1=head1
    node2=head2
    while node1 and node2:
        tmp1,tmp2=node1.next,node2.next
        node1.next=node2
        node1=tmp1
        node2.next=node1
        node2=tmp2
    return head1
c=merge(a,A)
dayin(c)






