#今天再次进行算法与数据结构的学习，今天得加快进度稍微
#关于堆排序再进行一轮,关于堆的学习，尽量一天一遍，然后加强理解
###这是大根堆

# li=[9,5,8,4,7]
# fun1(li,0,4)
# print(li)

# def fun(li):
#     n=len(li)
#     for i in range((n-2)//2,-1,-1):
#         fun1(li,i,n-1)
#     for i in range(n-1,-1,-1):
#         fun1(li,0,i-1)
#         li[i],li[0]=li[0],li[i]
#     print(li)





# fun(list,5)
###topk问题，然后建立小根堆的调整过程
# def sift(li,low,high):
#     tmp=li[low]
#     i=low
#     j=2*i+1
#     while j <= high:
#         if j+1 <= high and li[j+1]>li[j]:
#             j=j+1
#         if li[j]>tmp:
#             li[i]=li[j]
#             i=j
#             j=2*i+1
#         else:
#             break
#         li[i]=tmp

# def shixian(li,k):
#     li1=li[0:k]
#     xgd(li1,0,k-1)
#     for i in range(k,len(li)):
#         if li[i]>li1[1]:
#             li1[1]=li[i]
#             xgd(li1,0,k-1)
#     print(li1)
#
#
# list=[1,2,4,5,6,7,4,3,2,4,5,6,8,9,0,8,6,4]
# xgd([1,9,4,5,6],0,4)
# import heapq
# li=[9,5,8,4,7]
# print(li)
# sift(li,0,4)
# print(li)
# import heapq
# heapq.heapify(li)
# print(li)
#课程中关于最大最小堆的问题讲解存在错误，具体的写法，如下：
#首先需要值得强调的是，对于小根堆可以用内置函数，heapq来进行小根堆的生成，这里仅进行大根堆的函数定义
# import heapq
# def gg(li,k):
#     li1=li[0:k]
#     heapq.heapify(li1)
#     for i in range(k,len(li)):
#         if li[i]>li1[0]:
#             li1[0]=li[i]
#             heapq.heapify(li1)
#     print(li1)
#

# li=[2,3,54,62,1,1,3,5,6,7,90]
# # gg(li,5)
# #生成大根堆的函数定义来来来
# fun1(li,0,len(li)-1)
# print(li)
# def heap_adjust(L, start, end):
#     temp = L[start]
#     i = start
#     j = 2 * i
#     while j <= end:
#         if (j < end) and (L[j] < L[j + 1]):
#             j += 1
#         if temp < L[j]:
#             L[i] = L[j]
#             i = j
#             j = 2 * i
#         else:
#             break
#     L[i] = temp
#
#
# def build_heap(array, n, i):
#     largest = i
#     l = 2 * i + 1  # left = 2*i + 1
#     r = 2 * i + 2  # right = 2*i + 2
#
#     if l < n and array[i] < array[l]:
#         largest = l
#     if r < n and array[largest] < array[r]:
#         largest = r
#     if largest != i:
#         array[i], array[largest] = array[largest], array[i]
#         build_heap(array, n, largest)
#
#
# def Heap_Sort(array):
#     for i in range(len(array), -1, -1):
#         build_heap(array, len(array), i)
#     for i in range(len(array) - 1, 0, -1):
#         array[i], array[0] = array[0], array[i]
#         build_heap(array, i, 0)


def fun1(li,low,high):
    tmp=li[low]
    i=low
    j=2*i+1
    while j<=high:
        if j+1<=high and li[j+1]>li[j]:
            j=j+1
        if li[j]>tmp:
            li[i]=li[j]
            i=j
            j=2*i+1
        else:
            break
    li[i]=tmp




#对于目前我所提出的两个例子，显然没有得到解决
# list1=[1,9,4,5,7]#小根堆
#
# list2=[9,5,8,4,7]#大根堆
# # print(list2)
# import heapq
# heapq._heapify_max(list2)   但是引用了库里的函数倒是可以进行解决，但是根本的算法还是没有进行解决，
# print(list2)

# fun1(list2,0,len(list2)-1)
# print(list2)
#利用库里的函数进行排序把
import heapq
def fun1(li):
    heapq.heapify(li)
    for i in range(1,len(li)):
        list = list[i:len(li)]
        heapq.heapify(list)
        li[i]=list[0]
    print(li)

li=[2,3,4,5,2,45,6,3]






