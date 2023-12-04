#关于Transformer的程序测试及其详读
#....
#算法与数据结构
#插入排序算法采用for循环来进行
# def func(list):
#     for i in range(1,len(list)):
#         for j in range(i):
#             if list[i]<=list[j]:
#                 s=list[i]
#                 list[j+1:i+1]=list[j:i]
#                 list[j]=s
#     print(list)
# func([1,3,2,3,1,2,5])





# 快速排序重点是快快快
def fun1(data,left,right):
    a=data[left]
    while left < right:
        for i in range(right,left-1,-1):
            if data[i]>a:
                right=i-1
            if data[i]<=a:
                s=data.index(a)
                data[i],data[s]=data[s],data[i]
                right=i-1
                break
        for j in range(left,right+1):
            if data[j]<a:
                left=j+1
            if data[j]>=a:
                s=data.index(a)
                data[j],data[s]=data[s],data[j]
                left=j+1
                break
    return data.index(a)
def fun2(data,left,right):
    if left<right:
        mid=fun1(data,left,right)
        fun2(data,left,mid-1)
        fun2(data,mid+1,right)
    return data
data=[2,0,2,1,1,0]
print(fun2(data,0,len(data)-1))
