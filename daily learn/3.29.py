#今天堆的最后一步理解
#1.生成堆，本身不是堆，但是左右都是堆，具体到三个数显然成立
# def fun1(list,low,high):
#     i=low
#     j=2*i+1
#     tmp=list[low]
#     while j<=high:
#         if j+1 <= high and list[j]<list[j+1]:
#             j=j+1
#         if list[j]>tmp:
#             list[i]=list[j]
#             i=j
#             j=2*i+1
#         else:
#             break
#     list[i]=tmp
# def fun2(list):
#     n=len(list)
#     for i in range((n-2)//2,-1,-1):
#         fun1(list,i,n-1)
#     for i in range(n-1,-1,-1):
#         list[0],list[i]=list[i],list[0]
#         fun1(list,0,i-1)
#     print(list)
# list=[1,2,4,3,2,4,5,33,4,5,6,4,3,2]
# fun2(list)
#关于希尔排序
#类似与插入排序，等后期复习再进行一般的复习
#关于计数排序
# def fun(list,max):
#     li=[0]*(max+1)
#     for i in list:
#         li[i]+=1
#     list.clear()
#     for inx,val in enumerate(li):
#         for i in range(val):
#             list.append(inx)
#     print(list)
# list=[3,5,1,2,1,3,3,45,6,67,4,33,23,12,2,3,5,3,43]
# fun(list)
# #桶排序
#先分桶，然后对每个桶进行计数排序，1000之内，三个桶，0-300，300-600，600-1000
# def tong(list,s1,s2,s3):
#     li1=[]
#     li2=[]
#     li3=[]
#     for i in list:
#         if i>0 and i<=s1:
#             li1.append(i)
#         if i>s1 and i<=s2:
#             li2.append(i)
#         if i>s2 and i<s3:
#             li3.append(i)
list=[[]]*21
print(list)
