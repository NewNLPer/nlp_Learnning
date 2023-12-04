# -*- coding:utf-8 -*-
# @Time      :2022/6/28 10:30
# @Author    :Riemanner
# Write code with comments !!!
# def paidong(nums:list):
#     if len(nums)==1:
#         return nums
#     else:
#         nums.sort(reverse=True)
#         n=len(nums)
#         s1=n//2###编辑的开头
#         s2=s1+1###插入的开头
#         s3=0
#         while len(nums)<n+s1:
#             nums.insert(s2,nums[s3])
#             s2+=2
#             s3+=1
#         while len(nums)>n:
#             nums.pop(0)
#         return nums
# print(paidong(nums = [1,0]))
def wiggleMaxLength(nums):
    if len(nums)==1:
        return 1
    elif len(nums)==2 and nums[0]==nums[1]:
        return 1
    else:
        while len(nums)!=1 and nums[0]==nums[1]:
            nums.pop(0)
        if len(nums)<=2:
            return len(nums)
        else:
            n=len(nums)
            dp=[[0]*3 for _ in range(n)]
            dp[0][0]=1
            dp[0][1]=nums[0]
            dp[1][0]=2
            dp[1][1]=nums[1]
            dp[1][2]=(nums[1]-nums[0])//abs(nums[1]-nums[0])
            for i in range(2,n):
                if nums[i]-dp[i-1][1]==0:
                    dp[i] = dp[i - 1]
                elif (nums[i]-dp[i-1][1])//abs(nums[i]-dp[i-1][1])*dp[i-1][2]==-1:
                    dp[i][0]=dp[i-1][0]+1
                    dp[i][1]=nums[i]
                    dp[i][2]=(nums[i]-dp[i-1][1])//abs(nums[i]-dp[i-1][1])
                else:
                    dp[i][0]=dp[i-1][0]
                    if dp[i-1][2]==1:
                        dp[i][1] = max(dp[i - 1][1], nums[i])
                    elif dp[i-1][2]==-1:
                        dp[i][1]=min(dp[i - 1][1], nums[i])
                    dp[i][2]=dp[i-1][2]
            return dp[-1][0]

def lengthOfLIS(nums):
    n = len(nums)
    dp=[[0]*2 for _ in range(n)]
    dp[0][0] = 1
    dp[0][1] = nums[0]
    for i in range(1, n):
        if nums[i] < dp[i - 1][1]:
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = min(dp[i - 1][1], nums[i])
        elif nums[i]==dp[i-1][1]:
            dp[i] = dp[i - 1]
        else:
            dp[i][0] = dp[i - 1][0] + 1
            dp[i][1] = nums[i]
    return dp[-1][0]



def maxUncrossedLines(nums1,nums2):
    n1,n2=len(nums1),len(nums2)
    dp=[[0]*n2 for _ in range(n1)]
    if nums1[0]==nums2[0]:
        dp[0][0]=1
    for i in range(n1):
        for j in range(n2):
            if i==0 and j==0:
                continue
            elif i==0:
                if dp[0][0]==1:
                    dp[i][j]=1
                else:
                    if nums1[i] in set(nums2[:j+1]):
                        dp[i][j]=1
                    else:
                        dp[i][j]=0
            elif j==0:
                if dp[0][0]==1:
                    dp[i][j]=1
                else:
                    if nums2[j] in set(nums1[:i+1]):
                        dp[i][j]=1
                    else:
                        dp[i][j]=0
            else:
                if nums1[i]==nums2[j]:
                    dp[i][j]=dp[i-1][j-1]+1
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    return dp[-1][-1]

def maxSubArray(nums):
    n=len(nums)
    dp=[0]*n
    res=nums[0]
    dp[0]=nums[0]
    for i in range(1,n):
        dp[i]=max(dp[i-1]+nums[i],nums[i])
        res=max(res,dp[i])
    return res


def countSubstrings(s):
    n1=len(s)
    res=0
    dp=[[0]*n1 for _ in range(n1)]
    for i in range(n1):
        dp[i][i]=1
        res+=1
    for i in range(n1-1,-1,-1):
        for j in range(i,n1):
            if i==j:
                continue
            elif s[i]==s[j]:
                if j-i==1:
                    dp[i][j]=1
                    res+=1
                else:
                    dp[i][j]=dp[i+1][j-1]
                    if dp[i][j]==1:
                        res+=1
    return res
print(countSubstrings('aaaaa'))
