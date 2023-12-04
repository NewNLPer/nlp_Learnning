#####今天进行链表的学习




# s=1
# pp=a
# while pp.next!=None:
#     s+=1
#     pp=pp.next
# print(s)
# print(a.item)

###头插法
# def toucha(li):
#     head=Node(li[0])
#     for element in li[1:]:
#         node=Node(element)
#         node.next=head
#         head=node
#     return head
# head=toucha([1,2,3,4])
###尾插法
# def weicha(list):
#     tail=Node(list[0])
#     for i in list[1:]:
#         node=Node(i)
#         tail.next=node
#         tail=node
#     return tail
# ss=weicha([1,2,3,4])
# print(ss.item)



class Node:### 定义节点
    def __init__(self,item):
        self.item= item
        self.next= None
a=Node(1)
b=Node(2)
c=Node(3)
d=Node(4)
a.next=b
b.next=c
c.next=d
# print(dayin(a),end=' ')
# def removeNthFromEnd(head,n):
#     #####先计算一下链表的长度
#     lena = 1
#     head1 = head  ##头节点的复制
#     while head1.next != None:
#         lena += 1
#         head1 = head1.next
#     if lena==n:
#         head=head.next
#         return head
#     else:
#         head2 = head
#         for i in range(lena-n-1):  ###找到删除元素的前一个，也就是p.next=目标元素
#             head2 = head2.next
#     head2.next = head2.next.next
#     return head
# cc=removeNthFromEnd(a,1)
# print('========================')
# print(dayin(cc),end=' ')
#
# print('===================++++++++++++++++++++++')
# c=Node()
# print(c)











class Node:### 定义节点
    def __init__(self,item):
        self.item= item
        self.next= None
a=Node(1)
b=Node(2)
c=Node(3)
d=Node(4)
a.next=b
b.next=c
c.next=d

def dayin(head):
    while head!=None:
        print(head.item,end=' ')
        head=head.next
dayin(a)
print()
print('=================================')


def swapPairs(head):
    if not head or not head.next:
        return head
    newHead = head.next
    cc=head.next.next
    newHead.next = head
    head.next = swapPairs(cc)
    return newHead
cc=swapPairs(a)
dayin(cc)
print()
print('================================')

def fun(nums):
    if len(nums)==0 or len(nums)==1:
        return nums
    nums=[nums[1],nums[0]]+fun(nums[2:])
    return nums
print(fun([1,2,3,4,5,6]))














