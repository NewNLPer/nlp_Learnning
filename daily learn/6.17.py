# -*- coding:utf-8 -*-
# @Time      :2022/6/17 17:35
# @Author    :Riemanner
# Write code with comments !!!
# def function(s1,s2):
#     k=(s2[1]-s1[1])/(s2[0]-s1[0])
#     b=s1[1]-k*s1[0]
#     return (k,b)
# def maxPoints(points):
#     dic_c={}
#     res=0
#     for each in points:###处理当k=inf时候的情况
#         dic_c[each[0]]=dic_c.get(each[0],0)+1
#         res=max(res,dic_c[each[0]])
#     for i in range(len(points)-1):
#         for j in range(i+1,len(points)):
#             if points[i][0]==points[j][0]:
#                 continue
#             res1=2
#             path = function(points[i], points[j])
#             for k in range(j+1,len(points)):
#                 if points[i][0]==points[k][0]:
#                     continue
#                 if function(points[i],points[k])==path:
#                     res1+=1
#             res=max(res,res1)
#     return res
# print(maxPoints([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]))
# from functools import cache
# def canPartition(nums):
#     nums.sort()
#     k=sum(nums)//2
#     if sum(nums)%2!=0 or nums[-1]>k:
#         return False
#     else:
#         @cache
#         def bt(path,k,start):
#             if start>=len(nums):
#                 return False
#             if path==k:
#                 return True
#             for i in range(start,len(nums)):
#                 if path+nums[i]<=k and bt(path+nums[i],k,i+1):
#                         return True
#             return False
#         return bt(0,k,0)
# def twoSum(nums,target):
#     dic_c={}
#     for i in range(len(nums)):
#         dic_c[nums[i]]=dic_c.get(nums[i],[])+[i]
#     for key in dic_c:
#         if target-key==key and len(dic_c[key])>=2:
#             return dic_c[key][:2]
#         elif target-key==key and len(dic_c[key])==1:
#             continue
#         elif dic_c.get(target-key,False):
#             return dic_c[key][:1] + dic_c[target - key][:1]
# print(twoSum(nums = [3,2,4], target = 6))
def maximalSquare(matrix):
    n=len(matrix)
    m=len(matrix[0])
    t=min(m,n)
    while t>0:
        for i in range(0,n-t+1):
            for j in range(0,m-t+1):
                board=matrix[i:i+t]
                for k in range(len(board)):
                    if set(board[k][j:j+t])!={'1'}:
                        res=(k+i,board[k].index('0'))
                        break
                else:
                    return t*t
                if j<=res[1] or i <=res[0]:
                    continue
        t-=1
    return 0
# def originalDigits(s):
#     dic_C={}
#     for i in s:###统计元素
#         dic_C[i]=dic_C.get(i,0)+1
#     s1=['','','','','','','','','','']
#     s2=[('z',0,'zero'),
#         ('w',2,'two'),
#         ('u',4,'four'),
#         ('f',5,'five'),
#         ('x',6,'six'),
#         ('v',7,'seven'),
#         ('g',8,'eight'),
#         ('r',3,'three'),
#         ('o',1,'one'),
#         ('i',9,'nine')]
#     for touple1 in s2:
#         if dic_C.get(touple1[0],0)!=0:
#             ans=dic_C.get(touple1[0])
#             s1[touple1[1]]=str(touple1[1])*ans
#             for alph in touple1[2]:
#                 # if dic_C.get(alph,0)!=0:
#                 dic_C[alph]-=ans
#
#     return ''.join(s1)
# print(originalDigits("ereht"))
# def function(target):
#     res=0
#     for i in range(1,target-1):
#         res+=i
#     return res
# def numberOfArithmeticSlices(nums):
#     if len(nums)<=2:
#         return 0
#     else:
#         start_zhen=2
#         zhuizhong=2
#         res=0
#         while start_zhen<len(nums):
#             if nums[start_zhen]-nums[start_zhen-1]==nums[start_zhen-1]-nums[start_zhen-2]:
#                 start_zhen+=1
#                 zhuizhong+=1
#             else:
#                 res+=function(zhuizhong)
#                 start_zhen+=1
#                 zhuizhong=2
#         res+=function(zhuizhong)
#     return res
# print(numberOfArithmeticSlices(nums = [1,2,3,4,6,7,8,9,10]))
# def reconstructQueue(people):
#     dic_c={}
#     res=0
#     list=[]
#     for each in people:
#         dic_c[each[1]]=dic_c.get(each[1],[])+[each]
#         res=max(res,each[1])
#     for keys in dic_c:
#         dic_c[keys].sort()
#     list+=dic_c[0]
#     for i in range(1,res+1):
#         if dic_c.get(i,0)==0:
#             continue
#         else:
#             for dui in dic_c[i]:
#                 start_zhen=0
#                 res1=0
#                 while res1<i and start_zhen<=len(list)-1:
#                     if dui[0]<=list[start_zhen][0]:
#                         start_zhen+=1
#                         res1+=1
#                     else:
#                         start_zhen+=1
#                 while start_zhen<=len(list)-1 and list[start_zhen][0]<dui[0]:
#                     start_zhen+=1
#                 list.insert(start_zhen,dui)
#     return list
# print(reconstructQueue())
# def canFinish(numCourses, prerequisites):
#     if len(prerequisites)==0:
#         return [0]
#     else:
#         du1={}
#         du2=set()
#         for each in prerequisites:
#             if each[0] in du1:
#                 du1[each[0]].add(each[1])
#                 du2.add(each[0])
#                 du2.add(each[1])
#             else:
#                 du1[each[0]]={each[1]}
#                 du2.add(each[0])
#                 du2.add(each[1])
#         for i in du2:
#             if not du1.get(i,False):
#                 du1[i]=({})
#         res=[]
#         nextCourse = set(list(range(numCourses)))
#         for i in nextCourse:
#             if i not in du2:
#                 res.append(i)
#         ans=0
#         while ans==0:
#             ans=1
#             res1 = set()
#             for keys in du1:
#                 if len(du1[keys])==0:
#                     res1.add(keys)
#                     res.append(keys)
#                     du1[keys]={'*'}
#             for items in res1:
#                 for key in du1:
#                     if key in res1:
#                         continue
#                     else:
#                         du1[key].discard(items)
#                         if len(du1[key])==0:
#                             ans=0
#         if len(res)==numCourses:
#             return res
#         else:
#             return []
# print(canFinish(3,[[1,0],[1,2],[0,1]]))

def cfunction(n,k):
    def jiecheng(n):
        res = 1
        for i in range(1,n+1):
            res *= i
        return res
    return jiecheng(n)//(jiecheng(k)*jiecheng(n-k))
def threeSumMulti(arr,target):
    dic_c={}
    for i in arr:
        dic_c[i]=dic_c.get(i,0)+1
    nums=list(set(arr))
    nums.sort()
    nums1=set(nums)
    ###无重复数组的计算，双指针
    ans=0
    for i in range(len(nums)-2):
        ans1=0
        start_zhen=i+1
        end_zhen=len(nums)-1
        while start_zhen<end_zhen:
            if nums[i]+nums[start_zhen]+nums[end_zhen]<target:
                start_zhen+=1
            elif nums[i]+nums[start_zhen]+nums[end_zhen]>target:
                end_zhen-=1
            else:
                ans1+=(dic_c[nums[i]]*dic_c[nums[start_zhen]]*dic_c[nums[end_zhen]])
                start_zhen+=1
                end_zhen-=1
        ans+=ans1
    ###有重复数组的计算
    for num in nums:
        if dic_c[num]>=2:
            if target-2*num in nums1 and num!=target-2*num:
                ans+=cfunction(dic_c[num],2)*dic_c[target-2*num]
            elif target==3*num:
                ans+=cfunction(dic_c[num],3)
    return ans
def findAndReplacePattern(words,pattern):
    res=[]
    for word in words:
        if len(word)!=len(pattern):
            continue
        else:
            dic_c={}
            yin=set()###被映射过的记录
            c=0
            for i in range(len(word)):
                if word[i] not in dic_c and pattern[i] not in yin:
                    dic_c[word[i]]=pattern[i]
                    yin.add(pattern[i])
                elif word[i] in dic_c and dic_c[word[i]]!=pattern[i]:
                    c=1
                    break
                elif word[i] in dic_c and dic_c[word[i]]==pattern[i]:
                    continue
                elif pattern[i] in yin:
                    c=1
                    break
            if c==0:
                res.append(word)
    return res


class Treenode:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
a=Treenode(1)
b=Treenode(2)
c=Treenode(3)
d=Treenode(4)
e=Treenode(5)

a.left=b
a.right=c
c.left=d
d.right=e

# def sumNumbers1(root):
#     res=[]
#     def bt(root,path):
#         if not root:
#             return
#         if not root.left and not root.right:
#                 res.append(path+[root.val])
#         bt(root.left,path+[root.val])
#         bt(root.right,path+[root.val])
#     bt(root,[])
#     return res
#
#
# def findMaxAverage(nums,k):
#     if k==1:
#         return float(max(nums))
#     else:
#         start_zhe=0
#         end_zhen=start_zhe+k-1
#         res1=sum(nums[start_zhe:end_zhen+1])
#         res=sum(nums[start_zhe:end_zhen+1])
#         while end_zhen<len(nums)-1:
#             start_zhe+=1
#             end_zhen=start_zhe+k-1
#             res1=res1+nums[end_zhen]-nums[start_zhe-1]
#             res=max(res,res1)
#         return res/k
# print(findMaxAverage([1,12,-5,-6,50,3],4))



def findSubstring(s,words):
    dic_c1={}
    res=[]
    first=set()
    seconde=set()
    for word in words:
        dic_c1[word]=dic_c1.get(word,0)+1
        first.add(word[0])
        seconde.add(word)
    start_zhen=0
    while start_zhen<len(s)-len(words)+1:
        if s[start_zhen:start_zhen+len(words)]==''.join(words):
            res.append(start_zhen)
            start_zhen+=1
        else:
            dic_c2 = {}
            if s[start_zhen] in first and s[start_zhen:start_zhen+len(words[0])] in seconde:###首先先找到
                dic_c2[s[start_zhen:start_zhen+len(words[0])]]=1
                end_zhen=start_zhen+len(words[0])
                while end_zhen<=len(s):
                    if sum(dic_c2.values())<=sum(dic_c1.values()) and dic_c1==dic_c2:
                        res.append(start_zhen)
                        start_zhen+=1
                        break
                    elif end_zhen==len(s):
                        start_zhen+=1
                        break
                    elif dic_c1.get(s[end_zhen:end_zhen+len(words[0])],False) and dic_c2.get(s[end_zhen:end_zhen+len(words[0])],0)+1<=dic_c1[s[end_zhen:end_zhen+len(words[0])]]:
                        dic_c2[s[end_zhen:end_zhen+len(words[0])]]=dic_c2.get(s[end_zhen:end_zhen+len(words[0])],0)+1
                        end_zhen+=len(words[0])
                    else:
                        start_zhen+=1
                        break
            else:
                start_zhen+=1
    return res


def eraseOverlapIntervals(intervals):
    dic_c1={}
    dic_c2={}
    res=0
    for interval in intervals:
        dic_c1[interval[0]]=dic_c1.get(interval[0],[])+[interval]
        if dic_c2.get(interval[0],False):
            dic_c2[interval[0]]=min(dic_c1[interval[0]],key=lambda x:x[1])[1]
        else:
            dic_c2[interval[0]]=interval[1]
    for i in range(max(dic_c1.keys())):
        if dic_c2.get(i,False):

        else:
            continue


print(eraseOverlapIntervals(intervals = [[1,2],[2,3],[3,4],[1,3]]))