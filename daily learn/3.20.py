#关于堆的继续相当于是背诵把
def fun(li,low,high):
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
def fun1(li):
    n=len(li)
    for i in range((n-2)//2,-1,-1):
        fun(li,i,n-1)
    for i in range(n-1,-1,-1):
        li[0],li[i]=li[i],li[0]
        fun(li,0,i-1)
    print(li)
list=[1,3,563,2,34,6,2,3,5,7,5,7,9,0,8,5]
fun1(list)