### 1.快排
def quick_sort(nums,left,right):
    def take_nums(nums,left,right):
        tmp=nums[left]
        while left<right:
            while left<right and nums[right]>=tmp:
                right-=1
            nums[left],nums[right]=nums[right],nums[left]
            while left<right and nums[left]<=tmp:
                left+=1
            nums[left],nums[right]=nums[right],nums[left]
        nums[left]=tmp
        return left
    if left<right:
        mid=take_nums(nums,left,right)
        quick_sort(nums,left,mid-1)
        quick_sort(nums,mid+1,right)
        return nums
### 2.top_k算法
def top_k(nums,k):
    def sift(nums,low,high):
        tmp=nums[low]
        i=low
        j=2*i+1
        while j<=high:
            if j+1<=high and nums[j+1]<nums[j]:
                j=j+1
            if nums[j]<tmp:
                nums[i]=nums[j]
                i=j
                j=2*i+1
            else:
                break
        nums[i]=tmp
    head=nums[:k]
    for i in range((k-2)//2,-1,-1): #实现堆的过程
        sift(head,i,k-1)
    for i in range(k,len(nums)): #遍历数组
        if nums[i]>head[0]:
            head[0]=nums[i]
            sift(head,0,k-1)
    for i in range(k-1,-1,-1):
        head[0],head[i]=head[i],head[0]
        sift(head,0,i-1)
    return head
# nums=[1,9,2,4,5,3,3,54,32,2]
# print(top_k(nums,5))
### 3.二分查找
def erfen(nums,target):
    start=0
    end=len(nums)-1
    while start<=end:
        mid=(start+end)//2
        if nums[mid]>target:
            end=mid-1
        elif nums[mid]<target:
            start=mid+1
        else:
            return mid
    return -1,start
### 4.二分查找重复元素
'''
其余的不变，仅相等时反方向的压缩，最后来个判断
'''
### 5.子集
def ziji(nums):
    res=[]
    def bt(start,path):
        res.append(path[:])
        for i in range(start,len(nums)):
            path.append(nums[i])
            bt(i+1,path)
            path.pop()
    bt(0,[])
    return res
### 6.排列
def paile(nums):
    res=[]
    n=len(nums)
    def bt(path,nums):
        if len(path)==n:
            res.append(path[:])
        for i in range(len(nums)):
            path.append(nums[i])
            bt(path,nums[:i]+nums[i+1:])
            path.pop()
    bt([],nums)
    return res
### 7.回文切割
def qiege(s):
    res=[]
    def bt(start,path):
        if start>=len(s):
            res.append(path[:])
        for i in range(start,len(s)):
            p=s[start:i+1]
            if p==p[::-1]:
                path.append(p)
            else:
                continue
            bt(i+1,path)
            path.pop()
    bt(0,[])
    return res
print(qiege('aasda'))
### 8.链表
### 9.二叉树
### 10.动态规划

