#1。利用库里的调用函数，大根堆与小根堆，来进行排序函数的定义
#2.传统的大根堆与小根堆网上的写法，（尽管不太对，但是还是写一下把）
# import heapq
# ##大根堆的排序
# def fun(list):
#     li=[]
#     for i in range(0,len(list)):
#         heapq._heapify_max(list)
#         li.append(list[0])
#         list.pop(0)
#         print(li)

# list=[9,5,8,4,7]
#
#
# def fun2(list,low,high):
#     tmp=list[low]
#     i=low
#     j=2*i+1
#     while j <= high:
#         if j+1<=high and list[j+1]>list[j]:
#             j=j+1
#         if list[j]>tmp:
#             list[i]=list[j]
#             i=j
#             j=2*i+1
#         else:
#             break
#     list[i]=tmp
#     print(list)
# fun2(list,0,4)
# print('===================')
# def fun3(list):
#     n=len(list)
#     for i in range((n-2)//2,-1,-1):
#         fun2(list,i,n-1)
#     for i in range(len(list)-1,-1,-1):
#         list[0],list[i]=list[i],list[0]
#         fun2(list,0,i-1)
#     print(list)
# fun3(list)
# print(list)
#####对待大根堆与小根堆的学习到此结束，下面给出总结
'''
首先对于现在目前的大小根堆的底层算法要记住，虽然还存在不能解决的问题，但是可以先记住，然后利用这个算法进行排序
对于heaqp的调用，善于 使用，以及排序
'''
####归并排序的学习开始，但是每天还是要进行堆的算法的练习，对于目前的算法要每日练习争取记住
def merge(list,low,mid,high):
    i=low
    j=mid+1
    li=[]
    while i<=mid and j<=high:
        if list[i]<list[j]:
            li.append(list[i])
            i += 1
        else:
            li.append(list[j])
            j += 1
    while i<=mid:
        li.append(list[i])
        i+=1
    while j<=high:
        li.append(list[j])
        j+=1
    list[low:high+1]=li
    print(list)


def zuihou(list,low,high):
    if low<high:
        mid=(low+high)//2
        zuihou(list,low,mid)
        zuihou(list,mid+1,high)
        merge(list,low,mid,high)
        print(list)
    print('=======================')
    print(list)

list=[1,3,5,7,8,2,4,6]
zuihou(list,0,len(list)-1)

##############################
def fun(list,low,high):
    i=low
    j=2*i+1
    tmp=list[low]
    while j<=high:
        if j+1<high and list[j+1]>list[j]:
            j=j+1
        if list[j]>tmp:
            list[j]=list[i]
            i=j
            j=2*i+1
        else:
            break
def fun1(list):
    n=len(list)
    for i in range((n-2)//2,-1,-1):
        fun(list,i,n-1)
    for i in range(n-1,-1,-1):
        fun(list,0,i-1)
    print(list)
