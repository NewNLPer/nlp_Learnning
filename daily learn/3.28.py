#结束了一周的乱七八糟的事情，这周继续而且要加强训练
#依旧是关于大根堆的生成以及根据大根堆进行排序
# def fun1(li,low,high):
#     tmp=li[low]
#     i=low
#     j=2*i+1
#     while j <= high:
#         if j+1 <= high and li[j+1]>li[j]:
#             j=j+1
#         if li[j]>tmp:
#             li[i],li[j]=li[j],li[i]
#             i=j
#             j=2*i+1
#         else:
#             break
#     li[i]=tmp
# wc
def fun1(list,low,high):
    i=low
    j=2*i+1
    tmp=list[low]
    while j<=high:
        if j+1<=high and list[j+1]>list[j]:
            j=j+1
        if list[j]>tmp:
            list[i]=list[j]
            i=j
            j=2*i+1
        else:
            break
    list[i]=tmp
def fun(list):
    n=len(list)
    for i in range((n-2)//2,-1,-1):
        fun1(list,i,n-1)
    print(list)

list=[9,5,8,4,7]
fun(list)