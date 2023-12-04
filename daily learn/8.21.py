# -*- coding:utf-8 -*-
# @Time      :2022/8/21 9:51
# @Author    :Riemanner
# Write code with comments !!!
def isPrefixOfWord(sentence: str, searchWord: str) -> int:
    res=sentence.split(' ')
    for i in range(len(res)):
        if len(res[i])>=len(searchWord):
            start_zhen=0
            while start_zhen<len(searchWord):
                if res[i][start_zhen]==searchWord[start_zhen]:
                    start_zhen+=1
                else:
                    break
            if start_zhen==len(searchWord):
                return i+1
    return -1


def minNumberOfHours(initialEnergy: int, initialExperience: int, energy: list[int], experience: list[int]) -> int:
    res=0
    start_zhen=0
    while start_zhen<len(energy):
        if initialExperience>experience[start_zhen]:
            initialExperience+=experience[start_zhen]
        else:
            res+=experience[start_zhen]-initialExperience+1
            initialExperience+=(experience[start_zhen]-initialExperience+1+experience[start_zhen])
        if initialEnergy>energy[start_zhen]:
            initialEnergy-=energy[start_zhen]
            start_zhen+=1
        else:
            res+=(energy[start_zhen]-initialEnergy+1)
            initialEnergy+=(energy[start_zhen]-initialEnergy+1-energy[start_zhen])
            start_zhen+=1
    return res


def largestPalindromic(num: str) -> str:
    dic_c={}
    dic_c1=[]#偶数组合长度
    dic_c2=[]#奇数组合长度
    s=set()
    for item in num:
        dic_c[item]=dic_c.get(item,0)+1
        s.add(item)
    for keys in dic_c:
        if dic_c[keys] % 2 == 0:
            dic_c1.append(dic_c[keys])
        else:
            dic_c2.append(dic_c[keys])
    res1=float('-inf')
    res3=''
    res2=list(s)
    res2.sort(reverse=True)
    for i in range(len(res2)):
        if dic_c[res2[i]]%2==0:
            if res3 or(not res3 and res2[i]!='0'):
                res3+=(str(res2[i])*(dic_c[res2[i]]//2))
        else:
            if dic_c[res2[i]]==1:
                res1=max(res1,int(res2[i]))
            else:
                if res3 or (not res3 and res2[i] != '0'):
                    res3+=(str(res2[i])*((dic_c[res2[i]]-1)//2))
                    res1=max(res1,int(res2[i]))
    if not res3 and res1==float('-inf'):
        return '0'
    elif len(dic_c2)==0 and res3:
        return str(int(res3+res3[::-1]))
    else:
        return str(int(res3+str(res1)+res3[::-1]))









