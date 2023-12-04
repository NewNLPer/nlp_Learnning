#按照顺序这是个非常有意思的函数说明
def func(x):
    if x>0:
        func(x-1)
        print(x)
func(3)
# def fun(li,ke):
#     for c,v in enumerate(li):
#         if v==ke:
#             print(c)
#             break
#     else:
#         print('没找到')

#冒泡排序及其改进 23步建议进行debug的检查，
# def fun(lis):
#     for i in range(len(lis)-1):
#         exchange = False
#         for j in range(len(lis)-i-1):
#             if lis[j]>lis[j+1]:
#                 lis[j],lis[j+1]=lis[j+1],lis[j]
#                 exchange = True
#                 print(exchange)
#         print(lis)
#         if not exchange:
#             return
# fun([9,10,4,5,6,7,8])
#插入排序
#简易版
# def fun(li1):
#     li2=[]
#     for i in range(len(li1)):
#         s=min(li1)
#         li2.append(s)
#         li1.remove(s)
#     print(li2)
# fun([1,5,2,6])

#准确版
# def fun(li1):
#     for i in range(len(li1)-1):
#         s=min(li1[i:len(li1)])
#         o=li1.index(s)
#         li1[i],li1[o] = li1[o],li1[i]
#         print(li1)
# fun([2,0,2,1,1,0])

def fun(list):
    for i in range(len(list)):
        for j in  range(i+1,len(list)):
            if list[j]<list[i]:
                list[i],list[j]=list[j],list[i]
    return list
fun([2,0,2,1,1,0])






























