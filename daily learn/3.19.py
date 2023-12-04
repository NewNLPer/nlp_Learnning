#昨天没有进行学习，今天继续进行算法与数据结构的学习，关于堆的学习
def sitf(li,low,high):
    i=low
    j=2*i+1
    tmp=li[low]
    while j<=high:
        if j+1<=high and li[j+1]>li[j]:
            j=j+1
        if li[j]>tmp:
            li[i]=li[j]
            i=j
            j=2*i+1
        else:
            break
    li[i] = tmp

def paixu(li):
    for i in range((len(li)-2)//2,-1,-1):
        sitf(li,i,len(li)-1)
    for i in range(len(li)-1,-1,-1):
        li[0],li[i]=li[i],li[0]
        sitf(li,0,i-1)
    print(li)


paixu([1,42,2,34,3,1,3,5,90])






#关于堆排序自己再写一遍
# def fun(li,low,high):
#     tmp=li[low]
#     i=low
#     j=2*i+1
#     while j<=high:
#         if j+1<=high and li[j+1]>li[j]:
#             j=j+1
#         if li[j]>tmp:
#             li[i],li[j]=li[j],li[i]
#             i=j
#             j=2*i+1
#         else:
#             break
#     li[i]=tmp
# def fun1(li):
#     n=len(li)
#     for i in range((n-2)//2,-1,-1):









import torch
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
# 方便理解，此处假设batch_size = 1
x_input = torch.randn(2, 3)   # 预测2个对象，每个对象分别属于三个类别分别的概率
print(x_input)
# 需要的GT格式为(2)的tensor,其中的值范围必须在0-2(0<value<C-1)之间。
x_target = torch.tensor([0, 2])  # 这里给出两个对象所属的类别标签即可，此处的意思为第一个对象属于第0类，第二个我对象属于第2类
print(x_target)
loss = loss_fn(x_input, x_target)
print('loss:\n', loss)

n=3
dp=[['*']*n for _ in range(n)]
print(dp)