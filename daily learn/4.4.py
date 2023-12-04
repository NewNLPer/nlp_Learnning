#关于，队列，栈，链表的学习
# class Stack:
#     def __init__(self):
#         self.stack=[]
#     def push(self,element):
#         self.stack.append(element)
#     def pop(self):
#         return self.stack.pop()
#     def get_top(self):
#         if len(self.stack)>0:
#             return self.stack[-1]
#         else:
#             return None
# #括号匹配问题
# #哈希表
# # #####栈的迷宫问题的实现,通过栈储存来进行实现，栈的特点就是，后进先出，回溯法
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
#     lambda x,y:(x+1,y),
#     lambda x,y:(x-1,y),
#     lambda x,y:(x,y-1),
#     lambda x,y:(x,y+1)
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
#        else:#否则走回头路，再去搜寻
#            # maze[nextNode[0]][nextNode[1]]=2
#            stack.pop()
#     else:
#         print('无路可走')
#         return False
# maze_path(1,1,8,8)
#
# class Node:
#     def __init__(self,name,type='dir'):
#         self.name=name
#         self.type=type
#         self.children=[]
#         self.parent=None
# 模拟二叉树
from collections import deque
class BiTresNode:
    def __init__(self,data):
        self.data=data
        self.lchild=None
        self.rchild=None
a=BiTresNode('A')
b=BiTresNode('B')
c=BiTresNode('C')
d=BiTresNode('D')
e=BiTresNode('E')
f=BiTresNode('F')
g=BiTresNode('G')

e.lchild =a
e.rchild=g
a.rchild=c
c.lchild=b
c.rchild=d
g.rchild=f
root=e
#
# def pre_order(root):#前序遍历
#     if root:####如果不是空，与while true的比较
#         print(root.data,end=' ')
#         pre_order(root.lchild)
#         pre_order(root.rchild)
# pre_order(root)
# def in_order(root):#中序遍历
#     if root:
#         in_order(root.lchild)
#         print(root.data,end=',')
#         in_order(root.rchild)
# # in_order(root)
# def post_order(root):
#     if root:
#         post_order(root.lchild)
#         post_order(root.rchild)
#         print(root.data, end=',')
# # post_order(root)
# def level_order(root):
#     queue=deque()
#     queue.append(root)
#     while len(queue)>0:
#         node=queue.popleft()
#         print(node.data,end=',')
#         if node.lchild:
#             queue.append(node.lchild)
#         if node.rchild:
#             queue.append(node.rchild)
# # level_order(root)
# #二叉搜索树的查询和输入
# class BSTNode:
#     def __init__(self,data):
#         self.data=data
#         self.lchild=None
#         self.rchild=None
#         self.parent=None
# class BST:
#     def __init__(self):
#         self.root=None
#     def insert(self,node,val):
#         if not None:
#             node=BSTNode(val)
#         elif val<node.data:
#             node.lchild=self.insert(node.lchild,val)
#
#         return node


# def fun(i,j,list):#保证两边的数最大
#     num=(j-i-1)*min(list[i],list[j])
#     for k in range(i+1,j):
#         num=num-list[k]
#     return num
# def trap(height):
#     num=0
#     i = 0
#     while i+1<=len(height)-1:  # i已经确定，下面来确定j
#         while height[i+1]>=height[i]:
#             i=i+1
#             if i+1>=len(height):
#                 return 0+num
#         for j in range(i+2,len(height)):
#             if height[j]>=height[i]:
#                 num=num+fun(i,j,height)
#                 i=j
#                 break
#         else:
#             height[i]=height[i]-1
#     return num
#
# print(trap([4,2,3]))
def searchInsert(nums,target):
    n = len(nums)
    if target > nums[-1]:
        return n
    elif target < nums[0]:
        return 0
    elif n == 1:
        return 0
    else:
        if nums[n // 2] > target:
            for i in range(n // 2):
                if nums[i] >= target:
                    return i
            return n//2
        elif nums[n // 2] == target:
            return n // 2
        elif nums[n // 2] < target:
            for i in range(n // 2, n):
                if nums[i] >= target:
                    return i
print(searchInsert([1,3,5,6],2))

