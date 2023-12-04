#今天进行链表于哈希表的学习
# class Node:
#     def __init__(self,item):
#         self.item = item
#         self.next = Node
# a=Node([1,2,3,4])
# b=Node(5)
# c=Node(6)
# a.next=b
# b.next=c
# print(a.next.item)

#创建链表的方法，头插法与尾插法
# def tou(li):
#     head=Node(li[0])
#     for element in li[1:]:
#         node=Node(element)
#         node.next=head
#         head=node
#     return head
# cc=tou([1,2,3,4,5])
# print(cc.next.item)
# 定义结点类
# 链表节点实现
# class SingleNode(object):
#     def __init__(self, item):
#         # item：存放元素
#         self.item = item
#         # next:标识下一个结点
#         self.next = None
#
#
# # 单链表的实现
# class SingleLinkList(object):
#     def __init__(self, node=None):
#         # head：首节点
#         self.head = node
#     # 遍历链表 travel()
#     def travel(self):
#         # 游标记录当前所在的位置
#         cur = self.head
#
#         while cur is not None:
#             print(cur.item)
#             cur = cur.next
#
#     # 头部增加结点add()
#     def add(self, item):
#         # 新节点存储新数据
#         node = SingleNode(item)
#         node.next = self.head
#         self.head = node
#
#     # 尾部增加结点append()
#     def append(self, item):
#         # 新节点存储新数据
#         node = SingleNode(item)
#
#         # 判断是否是空链表
#         if self.is_empty():
#             self.head = node
#         else:
#             cur = self.head
#             # 找到尾结点
#             while cur.next is not None:
#                 cur = cur.next
#
#             cur.next = node
#
#     # 指定位置增加结点：insert(pos, item)
#     def insert(self, pos, item):
#
#         # 头部增肌新结点
#         if pos <= 0:
#             self.add(item)
#         elif pos >= self.length():
#             self.append(item)
#         else:
#             # 游标
#             cur = self.head
#             # 计数
#             count = 0
#             # 新结点
#             node = SingleNode(item)
#
#             # 1、找到插入位置的前一个结点
#             while count < pos - 1:
#                 cur = cur.next
#                 count += 1
#
#             # 2、完成插入新结点
#             node.next = cur.next
#             cur.next = node
#
#     # 删除结点 remove(item)
#     def remove(self, item):
#         # 游标
#         cur = self.head
#         # 辅助游标
#         pre = None
#
#         while cur is not None:
#             # 找到了要删除的元素
#             if cur.item == item:
#                 # 要删除的元素在头部
#                 if cur == self.head:
#                     self.head = cur.next
#                 else:
#                     pre.next = cur.next
#                 return
#             # 没有找到要删除的元素
#             else:
#                 pre = cur
#                 cur = cur.next
#
#     # 查找节点是否存在
#     def search(self, item):
#         # 游标
#         cur = self.head
#
#         while cur is not None:
#             # 找到了指定结点
#             if cur.item == item:
#                 return True
#             cur = cur.next
#
#         return False
#
# if __name__ == '__main__':
#     # 节点
#     node1 = SingleNode(10)
#     print(node1.item)
#     print(node1.next)
#     # 链表
#     link1 = SingleLinkList()
#     print(link1.head)
#     link2 = SingleLinkList(node1)
#     print(link2.head.item)
#     # 判空
#     print(link1.is_empty())
#     print(link2.is_empty())
#     # 长度
#     print(link1.length())
#     print(link2.length())
#     # 遍历
#     link2.travel()
#     # 头部增加结点
#     print("头部添加结点后")
#     link2.add(9)
#     link2.travel()
#     # 尾部增加结点
#     print("尾部添加结点后")
#     link2.append(11)
#     link2.travel()
#     # 指定位置增加结点
#     print("指定位置增加结点")
#     link2.insert(2, 0)
#     link2.travel()
#     # 删除结点
#     print("删除结点")
#     link2.remove(9)
#     link2.travel()
#     # 查找结点是否存在
#     print("查找结点是否存在")
#     print(link2.search(11))
#     print(link2.search(12))


# Definition for singly-linked list.
# class Solution:
#     def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
#         num1 = []
#         num2 = []
#         current1 = l1
#         current2 = l2
#         while current1 is not None:
#             num1.append(current1.val)
#             current1 = current1.next
#         while current2 is not None:
#             num2.append(current2.val)
#             current2 = current2.next
#         sumr = []
#         sign = 0
#         c = max(len(num1), len(num2))
#         while len(num1) != c:
#             num1.append(0)
#         while len(num2) != c:
#             num2.append(0)
#         for i in range(c):
#             if (sign == 0):
#                 sumr.append(num1[i] + num2[i])
#             if (sign == 1):
#                 sumr.append(num1[i] + num2[i] + 1)
#             if (sumr[i] >= 10):
#                 sumr[i] -= 10
#                 sign = 1
#             else:
#                 sign = 0
#         if (sign == 1):
#             sumr.append(1)
#         c = len(sumr)
#         head = ListNode()
#         cur = head
#         for i in range(c):
#             cur.next = ListNode(sumr[i])
#             cur = cur.next
#         return head.next

###先来计算一层的，然后层层递归
# def fun1(List):
#     num=0
#     i=0
#     while i != len(List)-1:
#         if List[i] != 1:
#             i+=1
#         if List[i]==1:
#             for j in range(i+1,len(List)):
#                 if List[j]==1:
#                     num=num+(j-i-1)
#                     i=j
#                     break
#                 if j==len(List)-1:
#                     i=j
#     return num
# def fun2(list):
#     for i in range(len(list)):
#         if list[i]>0:
#             list[i]=1
#         if list[i]<=0:
#             list[i]=0
#     return list
# class Solution:
#     def trap(self, height: List[int]) -> int:
#         s=max(height)
#         num=0
#         for i in range(s):
#             list1=[]
#             list1[:]= [x -i for x in height]
#             list2=fun2(list1)
#             num=num+fun1(list2)
#         return num
# def fun1(list):
#     num=0
#     i=0
#     a=sum(list)
#     j=len(list)-1
#     while i<j and a != 0:
#         while list[i]==0 and i<len(list):
#             i+=1
#         while list[j]==0 and j<len(list):
#             j-=1
#         if list[i] != 0 and list[j] != 0 and i<j:
#             c = min(list[i], list[j])
#             num=num+(j-i-1)*c
#         if i<j:
#             for k in range(i+1,j):
#                 if list[k]>=c:
#                     num=num-c
#                 if list[k]<c:
#                     num=num-list[k]
#             for u in range(i,j+1):
#                 list[u]=list[u]-c
#                 if list[u]<0:
#                     list[u]=0
#             if list[i]==0:
#                 i+=1
#             if list[j]==0:
#                 j-=1
#             a=sum(list[i:j+1])
#     print(num)
#
# fun1([2,1,0,2])
# def fun(s):
#     sss = []
#     ss = ''
#     i = 0
#     if len(s)==0:
#         print(0)
#     if len(s) != 0:
#         while i < len(s) and len(s) !=0 :
#             while s[i] not in ss and i <len(s):
#                 ss = ss + s[i]
#                 i += 1
#                 if s[i] in ss:
#                     i=i
#                     if i>=len(s):
#                         break
#             sss.append(len(ss))
#             ss=''
#         print(max(sss))
# fun('dvdffg')
def fun1(s):
    sss = []
    ss = ''
    i = 0
    if len(s) == 0:
        return 0
    if len(s) != 0:
        while i < len(s):
            while s[i] not in ss:
                ss = ss + s[i]
                i += 1
                if i >= len(s):
                    break
            sss.append(len(ss))
            ss = ''
        return max(sss)
def fun2(list):
    print(max(fun1(list),fun1(list[::-1])))
fun2('asjrgapa')



