#昨天那个快速排序算法仅仅是自己一个人的写法，但是对于标准的写法其时间复杂度还是比我的少，所以此类型的还得掌握
#quick_sort = lambda array: array if len(array) <= 1 else quick_sort([item for item in array[1:] if item <= array[0]]) + [array[0]] + quick_sort([item for item in array[1:] if item > array[0]])
#上面是其中的一个方法，不做过多的赘述
#下面给一个相当的解法 O（nlogn）
# def fun(li,left,right):
#     tmp=li[left]
#     while left<right:
#         while left<right and li[right]>=tmp:
#             right -= 1
#         li[left]=li[right]
#         while left<right and li[left]<=tmp:
#             left += 1
#         li[right]=li[left]
#     li[left]=tmp
#     return left

# from sklearn import preprocessing
# import numpy as np
# X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
# scaler= preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
# X_scaled = scaler.transform(X)
# print(X)
# print('==========================')
# print(X_scaled)
# print('===========================')
# X1=scaler.inverse_transform(X_scaled)
# print(X1)
# print('=============================')
# print(X1[0, -1])

def fun1(str):###正着开始进行
    c1=[]
    for i in range(1,len(str)-1):
        le1=1
        for j in range(i):
            if str[i] is not str[j]:
                le1=le1+1

            if str[i] is str[j]:
                break
        c1.append(le1)
        if str[i] is str[j]:
            break
    s=max(c1)
    return s
def fun2(str):###正着开始进行
    b=[]
    for i in range(len(str)-2,-1,-1):
        le=1
        for j in range(i+1,len(str)):
            if str[i] is not str[j]:
                le=le+1
            if str[i] is str[j]:
                break
        b.append(le)
        if str[i] is str[j]:
            break
    s=max(b)
    return s
def fun3(str):
    cc=max(fun1(str),fun2(str))
    print(cc)
c='dsbgjcda'
fun3(c)

