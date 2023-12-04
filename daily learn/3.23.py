###堆的写法，最后一遍把，算法题目重点刷力扣
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
    for i in range(n-1,-1,-1):
        list[i],list[0]=list[0],list[i]
        fun1(list,0,i-1)
    print(list)

li=[2,3,5,3,1,1,3,4,3,4,56,7,3,2,1,90]
fun(li)
