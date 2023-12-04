# #解决无重复的字符串与接水滴问题，关于链表、迷宫、类的总结
# def fun1(s):
#     sss = []
#     if len(s)==0:
#         return 0
#     if
#     for i in range(len(s)):
#         ss=''
#         ss=ss+s[i]
#         for j in range(i+1,len(s)):
#             if s[j] not in ss:
#                 ss=ss+s[j]
#                 continue
#             if s[j] in ss:
#                 break
#         sss.append(len(ss))
#     return max(sss)
# fun1('asjrgapa')
#截断处理函数,也就是两头的数都要比中间的数大
# def fun(i,j,list):
#     num=(j-i-1)*min(list[i],list[j])
#     for k in range(i+1,j):
#         num=num-list[k]
#     return num
# #接下来需要确定i，j的问题了
# def fun1(list):
#     num=0
#     i=0
#     while list[i+1]>=list[i]:
#         i+=1
#     for j in range(i+2,len(list)):
#         while list[j]<list[i] and list[j]>list[j-1]:

#
# def fun(nums):
#     S = []
#     k = 0
#     for i in range(len(nums)):
#         if nums[i] not in S:
#             S.append(nums[i])
#             k += 1
#         else:
#             continue
#     for j in range(k - 1):
#         S.append('_')
#     print(k)
#     print(S)
def fun(nums):
    S = []
    if len(nums) < 3:
        return []
    else:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if -(nums[i] + nums[j]) in nums[j + 1:] and {nums[i], nums[j], -nums[i] - nums[j]} not in S:
                    S.append({nums[i], nums[j], -nums[i] - nums[j]})
                    continue
    for i in range(len(S)):
        S[i] = list(S[i])
    print(S)

fun([0,0,1,0])
