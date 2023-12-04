#开始进行python的一些疑点的学习
#1.关于函数的def
# def add(a,b):
#     c=a+b
#     return a# 返回值
# print(add(2,4))
#递归函数的几个案例，注意理解
#阶乘的计算
# def fac(a):
#     if a==1:
#         return 1
#     else:
#         return a*fac(a-1)
# print(fac(3))
#斐波那契数列的计算
# def fff(b):
#     if b==1:
#         return 1
#     elif b==2:
#         return 1
#     else :
#         return fff(b-1)+fff(b-2)
# print(fff(1))
# print(fff(2))
# print(fff(5))
#总结起来就是需要对初值进行定义，然后自己调用自己的函数
#if  的用法
#其实就是普通的判断
'''
if :

elif:

....
else:

'''
# while 的用法
# x=0
# sum=0
# while x<101:
#     sum=sum+x
#     x+=1
# print(sum)

import math
###计算1-100之间的偶数和
# sum=0
# a=0
# while a<101:
#     sum=sum+a
#     a+=2
# print(sum)
# print('----------------------')
####水仙花数的确定
# list=[]
# x=range(100,1000)
# for i in x:
#     a=math.floor(i/100)
#     b=math.floor((i-100*a)/10)
#     c=i-100*a-10*b
#     if pow(a,3)+pow(b,3)+pow(c,3)==i:
#         list.append(i)
# print(list)
####break的学习，类似于银行密码，输入正确就不需要在进行输入了
# 满足if之后一旦执行break就会退出这个for循环或者while循环，后续不在进行下去，(跳出这个大的循环)
# 满足if之后一旦执行continue就会退出这个if循环,if之后的代码都不会再执行，然后紧接着下一个for循环或者while循环开始进行
# for i in range(3):
#     pwd=int(input('请输入您的密码:'))
#     if pwd ==888:
#         print('密码正确')
#         break
#     else:
#         print('密码错误，您还有%d次机会'%(2-i))
# break 与 while
# lis=[]
# x=range(1,51)
# for i in x:
#     if i%5!=0:
#         continue
#     lis.append(i)
# print(lis)
#####重头戏开始，关于面向对象和面向过程，类与对象的理解
# class student:
#     nativa='adasd'
#     def __init__(self,name,age,sex):
#         self.name=name
#         self.age=age
#         self.sex=sex
#     def eat(self):
#         print(self.name+'正在吃饭')
# stu1=student('dsad','20','1')
# stu2=student('asdad','23','1')
# print(stu2.name)
# print(student.nativa)
# print(stu1)



class Animal(object):
    '''
    猪和鸭子的基类（基因图纸表）
    '''
    def __init__(self, name): # 实例化的时候传入要制作的东西名字，如猪、鸭子
        self.name = name

    def makeMoth(self):
        #这里可以放其他制作细节
        print(self.name+'的嘴巴 制作完毕') #这里的self.name就是获取我们传入的name

    def makeEar(self):
        #这里可以放其他制作细节
        print(self.name+'的耳朵 制作完毕')

    def makeEye(self):
        #这里可以放其他制作细节
        print(self.name+'的眼睛 制作完毕')

    def makeHead(self):
        #这里可以放其他制作细节
        print(self.name+'的头 制作完毕')

    def makeBody(self):
        #这里可以放其他制作细节
        print(self.name+'的身体 制作完毕')

    def makeFoot(self):
        #这里可以放其他制作细节
        print(self.name+'的脚 制作完毕')

    def makeMerge(self):
        #这里可以放其他制作细节
        print(self.name+'合并完毕')

    def makeAll(self):
        # 一条龙。直接跑完整个流水线
        self.makeMoth()
        self.makeEar()
        self.makeEye()
        self.makeHead()
        self.makeBody()
        self.makeFoot()
        self.makeMerge()

class Duck(Animal):  #
    def makeMoth(self):
        #这里加详细的鸭子嘴巴制作流程，如长嘴巴，嘴巴硬
        print(self.name+'的嘴巴 制作完毕')

    def makeEar(self):
        #耳朵很小
        print(self.name+'的耳朵 制作完毕')

    def makeEye(self):
        #眼睛小小的
        print(self.name+'的眼睛 制作完毕')

    def makeHead(self):
        #头很小
        print(self.name+'的头 制作完毕')

    def makeBody(self):
        #略
        print(self.name+'的身体 制作完毕')

    def makeFoot(self):
        #略
        print(self.name+'的脚 制作完毕')

    def makeMerge(self):
        #略
        print(self.name+'合并完毕')

    def makeWing(self): #增加翅膀的制作流程
        #略
        print(self.name+'的翅膀 制作完毕')

    def makeAll(self): #因为增加了翅膀，所以要覆写这个函数
        self.makeMoth()
        self.makeEar()
        self.makeEye()
        self.makeHead()
        self.makeBody()
        self.makeFoot()
        self.makeWing()  #插入翅膀制作流程
        self.makeMerge()

stu=Duck('HHH')
stu.makeAll()


