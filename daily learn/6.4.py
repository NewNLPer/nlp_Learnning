# -*- coding:utf-8 -*-
# @Time      :2022/6/4 13:56
# @Author    :Riemanner
# Write code with comments !!!
class TreeNode():
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
a=TreeNode(1)
b=TreeNode(2)
c=TreeNode(2)
# d=TreeNode(1)
# e=TreeNode(3)
a.right=c
c.left=b

# def recoverTree(root):
#     """
#     Do not return anything, modify root in-place instead.
#     """
#     def zhong(root):
#         if not root:
#             return []
#         else:
#             return zhong(root.left)+[root]+zhong(root.right)
#     nums=zhong(root)
#     start_zhen=0
#     end_zhen=len(nums)-1
#     while start_zhen<end_zhen:
#         start_zhen1=0
#         end_zhen1=0
#         if max(nums[start_zhen:end_zhen+1],key=lambda x:x.val)==nums[end_zhen]:
#             end_zhen-=1
#             end_zhen1=1
#         if min(nums[start_zhen:end_zhen+1],key=lambda x:x.val)==nums[start_zhen]:
#             start_zhen+=1
#             start_zhen1=1
#         if start_zhen1==0 and end_zhen1==0:
#             break
#     nums[start_zhen].val,nums[end_zhen].val=nums[end_zhen].val,nums[start_zhen].val
#     return root
# print(recoverTree(a))


# import copy
# def isValidBST(root):
#     def zhong(root):
#         if not root:
#             return []
#         else:
#             return zhong(root.left)+[root.val]+zhong(root.right)
#     nums=zhong(root)
#     nums1=copy.deepcopy(nums)
#     nums1.sort()
#     if len(nums)!=len(list(set(nums))) or nums1!=nums:
#         return False
#     else:
#         return True
# print(isValidBST(a))


















###二叉树的遍历
# def sumNumbers1(root):
#     res=[]
#     def bt(root,path):
#         if not root:
#             return
#         if not root.left and not root.right:
#                 res.append(path+[str(root.val)])
#         bt(root.left,path+[str(root.val)])
#         bt(root.right,path+[str(root.val)])
#     bt(root,[])
#     return res
# print(sumNumbers1(a))

def countSegments(s):
    start_zhen=0
    res=[]
    while start_zhen<len(s):
        if not s[start_zhen].isalpha():
            start_zhen+=1
        else:
            end_zhen=start_zhen+1
            while end_zhen<len(s):
                if s[end_zhen].isalpha() or s[end_zhen] in [''] :
                    end_zhen+=1
                else:
                    res.append(s[start_zhen:end_zhen])
                    start_zhen=end_zhen
                    break
    return res
print(countSegments("Hello, my name is John*"))

"love live! mu'sic forever"