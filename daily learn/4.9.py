###力扣字
###只定义两个的，然后进行递归
# def function(s1,s2,dic):
#     S=[]
#     for i in range(len(dic[s1])):
#         for j in range(len(dic[s2])):
#             S.append(dic[s1][i]+dic[s2][j])
#     return S
# def letterCombinations(digits):
#     dic_c={'2':['a','b','c'],'3':['d','e','f'],'4':['g','h','i'],'5':['j','k','l'],'6':['m','n','o'],'7':['p','q','r','s'],'8':['t','u','v'],'9':['w','x','y','z']}
#     if len(digits)==0:
#         return []
#     elif len(digits)==1:
#         return dic_c[digits]
#     else:
#         n=len(digits)
#         if n==2:
#             print(function(digits[0],digits[1],dic_c))
#         elif n==3:
#             dic_c[digits[0:2]]=function(digits[0],digits[1],dic_c)
#             print(function(digits[0:2],digits[-1],dic_c))
#         elif n==4:
#             dic_c[digits[0:2]] = function(digits[0], digits[1], dic_c)
#             dic_c[digits[0:3]] = function(digits[0:2],digits[2],dic_c)
#             print(dic_c)
#             print(function(digits[0:3],digits[-1],dic_c))
# letterCombinations('5678')
s='(()*%d)'%2
print(s)


