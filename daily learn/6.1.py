# -*- coding:utf-8 -*-
# @Time      :2022/6/2 8:38
# @Author    :Riemanner
# Write code with comments !!!
class TreeNode():
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
a=TreeNode(1)
b=TreeNode(2)
c=TreeNode(3)
d=TreeNode(4)
e=TreeNode(5)

a.left=b
a.right=c
c.left=d
d.right=e
###当空姐点的也加入
# def cengxvbianli(root):
#     queue=[root]
#     res=[]
#     while queue.count(None)!=len(queue):
#         path=[]
#         ll=[]
#         for i in queue:
#             if i:
#                 path.append(i.val)
#             else:
#                 path.append('*')
#         res.append(path)
#         for node in queue:
#             if node==None:
#                 ll.append(None)
#                 ll.append(None)
#             else:
#                 if node.left:
#                     ll.append(node.left)
#                 else:
#                     ll.append(None)
#                 if node.right:
#                     ll.append(node.right)
#                 else:
#                     ll.append(None)
#         queue=ll
#     return res
# print(cengxvbianli(a))
# def checkInclusion(s1: str, s2: str) -> bool:
#     dic_c1={}
#     dic_c2={}
#     for i in s1:
#         dic_c1[i]=dic_c1.get(i,0)+1
#     for j in s2[:len(s1)]:
#         dic_c2[j]=dic_c2.get(j,0)+1
#     if dic_c1==dic_c2:
#         return True
#     for k in range(1,len(s2)-len(s1)+1):
#         if dic_c2[s2[k-1]]>=2:
#             dic_c2[s2[k-1]]-=1
#         else:
#             del dic_c2[s2[k-1]]
#         dic_c2[s2[k+len(s1)-1]]=dic_c2.get(s2[k+len(s1)-1],0)+1
#         if dic_c1==dic_c2:
#             return True
#     return False
# print(checkInclusion(s1 = "ab",s2 = "eidbaooo"))
def findUnsortedSubarray(nums):
    start_zhen=0
    end_zhen=len(nums)-1
    while start_zhen<end_zhen:
        max1=max(nums[start_zhen:end_zhen+1])
        min1=min(nums[start_zhen:end_zhen+1])
        start_zhen1=1
        end_zhen1=1
        if min1==nums[start_zhen]:
            start_zhen+=1
            start_zhen1=0
        if max1==nums[end_zhen]:
            end_zhen-=1
            end_zhen1=0
        if start_zhen1==1 and end_zhen1==1:
            return end_zhen-start_zhen+1
    return 0
print(findUnsortedSubarray(nums = [1,2,3,4]))