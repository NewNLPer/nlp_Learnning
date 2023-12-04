# -*- coding:utf-8 -*-
# @Time      :2022/5/7 9:37
# @Author    :Riemanner
# Write code with comments !!!
# 子集+组合+排列+桶排
# def ziji(nums):
#     res=[]
#     def bt(start,path):
#         if path not in res:
#             res.append(path[:])
#         for i in range(start,len(nums)):
#             path.append(nums[i])
#             bt(i+1,path)
#             path.pop()
#     bt(0,[])
#     return res
# print(ziji([1,2,2]))
# def zuhe(n,k):###n的k组合(无重复)
#     res=[]
#     def bt(path):
#         if len(path)==k:
#             res.append(path[:])
#         for i in range(1,n+1):
#             if i not in path:
#                 path.append(i)
#                 bt(path)
#                 path.pop()
#     bt([])
#     return res
# print(zuhe(4,2))
# def pailie(nums):##无重复元素
#     res=[]
#     def bt(path):
#         if len(path)==len(nums):
#             res.append(path[:])
#         for i in range(len(nums)):
#             if nums[i] not in path:
#                 path.append(nums[i])
#                 bt(path)
#                 path.pop()
#     bt([])
#     return res
# print(pailie([1,2,3]))
# def paile(nums):#有重复元素的组合
#     res=[]
#     def bt(nums,path):
#         if not nums:
#             res.append(path[:])
#         for i in range(len(nums)):
#             if i==0 or nums[i]!=nums[i-1]:
#                 path.append(nums[i])
#                 bt(nums[:i]+nums[i+1:],path)
#                 path.pop()
#     bt(nums,[])
#     return res
# print(paile([1,2,2,3]))
##############################解决#############################
#####关于正方形的桶
# def zhengfangxin(nums):
#     if sum(nums)%4!=0:
#         return False
#     c=sum(nums)//4
#     nums.sort(reverse=True)
#     if nums[0]>c:
#         return False
#     nums.sort()
#     k=4
#     while nums[-1]==c:
#         k-=1
#         nums.pop()
#     cc=[0]*k
#     def bt(inx):
#         if inx==len(nums):
#             return True
#         for i in range(k):
#             if cc[i]+nums[inx]<=c:
#                 cc[i]+=nums[inx]
#                 if bt(inx+1):
#                     return True
#                 cc[i]-=nums[inx]
#         return False
#     return bt(0)
# print(zhengfangxin([3,3,3,3,4]))
##最长连续序列的进行
# def zuichang(nums):
#     res=0
#     nums=set(nums)
#     while nums:
#         c=nums.pop()
#         k1=1
#         k2=0
#         qian=c-1
#         hou=c+1
#         while qian in nums:
#             nums.remove(qian)
#             k1+=1
#             qian-=1
#         while hou in nums:
#             nums.remove(hou)
#             k2+=1
#             hou+=1
#         res=max(res,k1+k2)
#     return res
# print(zuichang([3]))
# def nibolan(tokens):
#     pos={'*':lambda x,y:x*y,'+':lambda x,y:x+y,'-':lambda x,y:x-y,'/':lambda x,y:x/y}
#     cunchu_zhan=[]
#     for i in range(len(tokens)):
#         if tokens[i] not in {'+','-','*','/'}:
#             cunchu_zhan.append(int(tokens[i]))
#         else:
#             c1=cunchu_zhan[-1]
#             cunchu_zhan.pop()
#             c2=cunchu_zhan[-1]
#             cunchu_zhan.pop()
#             cunchu_zhan.append(int(pos[tokens[i]](c2,c1)))
#     return cunchu_zhan[-1]
# print(nibolan(tokens = ["2","1","+","3","*"]))
################开始进行记忆化的搜索的联系，包括备忘录的，加上装饰器的
######通过斐波那契数列为例
######普普通通的递归，重复的计算过程
# def function1(n):
#     if n==1 or n==2:
#         return 1
#     else:
#         return function1(n-1)+function1(n-2)
####进阶带装饰器的斐波那契，对计算过的数据进行缓存,可以看到速度得到了一个很好的提升
# from functools import lru_cache
# @lru_cache()
# def function2(n):
#     if n==1 or n==2:
#         return 1
#     else:
#         return function2(n-1)+function2(n-2)
# print(function2(100))
#####自带备忘录的斐波那契数列
# def function3(n):
#     hash_map = {1: 1, 2: 1}
#     def bt(n,hash_map):
#         if n in hash_map:
#             return hash_map[n]
#         else:
#             hash_map[n]=bt(n-1,hash_map)+bt(n-1,hash_map)
#             print(hash_map)
#             return hash_map[n]
#     return bt(n,hash_map)
# print(function3(200))
####回文字符串
# def zifuchaun(s):
#     res=[]
#     def bt(start,path):
#         if start>=len(s):
#             res.append(path[:])
#         for i in range(start,len(s)):
#             p=s[start:i+1]
#             if p==p[::-1]:
#                 path.append(p)
#                 bt(i+1,path)
#                 path.pop()
#     bt(0,[])
#     return res
# print(zifuchaun('aacva'))
# def func(s):
#     def bt(start,mome):
#         if start>=len(s):
#             return 1
#         if s[start]=='0':
#             return 0
#         if start in mome:
#             return mome[start]
#         a=bt(start+1,mome)
#         b=0
#         if int(s[start:start+2])<26 and len(s)-start>=2:
#             b=bt(start+2,mome)
#         mome[start]=a+b
#         return a+b

# def minimumTotal(triangle):
#     res = 0
#     for i in range(len(triangle)):
#         if i == 0:
#             res += min(triangle[i])
#             c=0
#         else:
#             res += min(triangle[i][c],triangle[i][c+1])
#             if triangle[i][c]>triangle[i][c+1]:
#                 c=c+1
#     return res
# print(minimumTotal([[-1],[2,3],[1,-1,-3]]))
# def minimumTotal(triangle):
#     for i in range(len(triangle)):
#         if i == 0:
#             min1 = triangle[i]
#         else:
#             min2 = []
#             for j in range(len(min1)):
#                 if i>=3 and j==len(min1)-1:
#                     min2.append(min1[j] + triangle[i][j])
#                 else:
#                     min2.append(min1[j] + triangle[i][j])
#                     min2.append(min1[j] + triangle[i][j + 1])
#             min1 = min2
#     return min(min1)
# print(minimumTotal(triangle = [[2],[3,4],[6,5,9],[4,4,8,0]]))
# def zifuchaun(s,t):
#     res=[]
#     def bt(start,path):
#         if len(path)>0:
#             res.append(path[:])
#         for i in range(start,len(s)):
#             if s[i] in t:
#                 path+=s[i]
#                 bt(i+1,path)
#                 path=path[:len(path)-1]
#     bt(0,'')
#     return res
# print(zifuchaun(s = "ADOECODEBANC", t = "ABC"))
# def bijiao(t1,t2):
#     for key in t1:
#         if t1.get(key)>t2.get(key,0):
#             return False
#     return True
# def fugai(s,t):
#     T=[]
#     T+=t
#     t_hash={}
#     for i in T:
#         if i not in t_hash:
#             t_hash[i]=1
#         else:
#             t_hash[i]+=1
#     res=[]
#     start_zhen=0
#     end_zhen=1
#     record_hash={s[start_zhen]:1,s[end_zhen]:1}#####记录走的的元素
#     while end_zhen<len(s)+1 and start_zhen<=len(s)-len(t):
#         if not bijiao(t_hash,record_hash):
#             end_zhen+=1
#             if end_zhen<len(s) and s[end_zhen] in record_hash:
#                 record_hash[s[end_zhen]]+=1
#             else:
#                 if end_zhen<len(s):
#                     record_hash[s[end_zhen]]=1
#         else:
#             if len(res)==0:
#                 res.append(s[start_zhen:end_zhen+1])
#                 start_zhen+=1
#                 record_hash[s[start_zhen]]-=1
#             else:
#                 if len(res[-1])>end_zhen-start_zhen:
#                     res[-1]=s[start_zhen:end_zhen]
#                     start_zhen+=1
#                     record_hash[s[start_zhen]]-=1
#                 else:
#                     start_zhen += 1
#                     record_hash[s[start_zhen]]-=1
#     if len(res)==0:
#         return ''
#     return res[-1]
# print(fugai(s = "ADOBECODEBANC", t = "ABC"))

# def bijiao(t1,t2):
#     for key in t1:
#         if t1.get(key)>t2.get(key,0):
#             return False
#     return True
#
# def func(s,t):
#     T=[]
#     T+=t
#     t_hash={}
#     for i in T:
#         if i not in t_hash:
#             t_hash[i]=1
#         else:
#             t_hash[i]+=1
#     res=[]
#     start_zhen=0
#     end_zhen=2
#     if s[start_zhen]!=s[end_zhen]:
#         record_hash={s[start_zhen]:1,s[end_zhen]:1}
#     else:
#         record_hash = {s[start_zhen]: 2}
#     while end_zhen<len(s)+1 and start_zhen<=len(s)-len(t):
#         if sum(record_hash.values())<sum(t_hash.values()) or t_hash.get(s[end_zhen],0)==0:
#             end_zhen+=1
#             if end_zhen<len(s):
#                 if s[end_zhen] in record_hash:
#                     record_hash[s[end_zhen]]+=1
#                 else:
#                     record_hash[s[end_zhen]]=1
#         elif not bijiao(t_hash,record_hash):
#             end_zhen+=1
#             if end_zhen<len(s):
#                 if s[end_zhen] in record_hash:
#                     record_hash[s[end_zhen]]+=1
#                 else:
#                     record_hash[s[end_zhen]]=1
#         else:
#             if len(res)==0:
#                 res.append(s[start_zhen:end_zhen])
#                 start_zhen+=1
#                 record_hash[s[start_zhen-1]]-=1
#             else:
#                 if len(res[-1])>end_zhen-start_zhen+1:
#                     res[-1]=s[start_zhen:end_zhen + 1]
#                     start_zhen+=1
#                     record_hash[s[start_zhen - 1]] -= 1
#                 else:
#                     start_zhen+=1
#                     record_hash[s[start_zhen - 1]] -= 1
#     if len(res)==0:
#         return ''
#     return res[-1]
# print(func('aa','a'))
# def singleNumber(nums):
#     nums.sort()
#     S = [nums[0]]
#     for i in range(1, len(nums)):
#         if len(S) == 1:
#             if S[-1] == nums[i]:
#                 S.pop()
#                 continue
#             else:
#                 return S[-1]
#         else:
#             S.append(nums[i])
#             if i==len(nums)-1:
#                 return S[0]
# print(singleNumber([1,1,2,3,3]))
# def singleNumber(nums):
#     """
#     :type nums: List[int]
#     :rtype: int
#     """
#     a = 0
#     for num in nums:
#         a = a ^ num
#     return a
# print(singleNumber([1,1,2,3,3]))


# def largestNumber(self, nums):
#     s = []
#     for i in nums:
#         if i >= 10:
#             s.append(i // 10)
#             s.append(i - (i // 10) * 10)
#         else:
#             s.append(i)
#     s.sort(reverse=True)
#     return ''.join(map(lambda x: str(x), s))
# print(largestNumber([10,2]))
def largestNumber(nums):
    if len(nums) == 1:
        return str(nums[0])
    elif len(nums) == 2:
        return str(max(int(str(nums[0]) + str(nums[1])), int(str(nums[1]) + str(nums[0]))))
    else:
        return str(max(int(str(nums[-1]) + str(largestNumber(nums[:len(nums)-1]))), int(str(largestNumber(nums[:len(nums)-1])) + str(nums[-1]))))
print(largestNumber(nums = [3,30,34,5,9,565,2,2,5,2,8,41,0,2,4,110,25,141,2,2,9,2,1,22,5]))
