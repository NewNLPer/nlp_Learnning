# -*- coding:utf-8 -*-
# @Time      :2022/5/12 10:17
# @Author    :Riemanner
# Write code with comments !!!
class Node():
    def __init__(self,val):
        self.val=val
        self.next=None
a=Node(1)
b=Node(2)
c=Node(3)
d=Node(4)
e=Node(5)
a.next=b
b.next=c
c.next=d
d.next=e
def dayin(head):
    while head:
        print(head.val,end=' ')
        head=head.next
##############################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def oddEvenList(head):
    if not head:
        return head
    odd = head
    even_head = even = head.next
    while odd.next and even.next:
        odd.next = odd.next.next
        even.next = even.next.next
        odd,even = odd.next,even.next
    odd.next = even_head
    return head













# def chongpailianbiao(head):
# ######快慢指针找中点######
#     fast_zhen=head
#     low_zhen=head
#     zuduan_zhen=head
#     while fast_zhen.next and fast_zhen.next.next:
#         fast_zhen=fast_zhen.next.next
#         low_zhen=low_zhen.next
#         zuduan_zhen=low_zhen.next
#     low_zhen.next=None
#
#
# ####开始进行反转####
#     pre,cur=None,zuduan_zhen
#     while cur:
#         tem=cur.next
#         cur.next=pre
#         pre=cur
#         cur=tem
# ###开始进行拼接######
#     node1,node2=head,pre
#     while node1 and node2:
#         tmp1,tmp2=node1.next,node2.next
#         node1.next=node2
#         node1=tmp1
#         node2.next=node1
#         node2=tmp2
#     return head
# c=chongpailianbiao(a)
# dayin(c)
