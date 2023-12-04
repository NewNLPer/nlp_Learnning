# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/3/13 19:24
coding with comment！！！
"""
import jieba
'''
目前大体的底层词汇增强方式如下
1.Softword
2.Ex-softword
3.Softlexicon
'''
###   1.Softword
s1='这座依山傍水的博物馆由国内一流的设计师主持设计的'
def softword(s:str):
    res=list(jieba.cut(s,cut_all=False))
    ans=[]
    for item in res:
        if len(item)==1:
            ans+=['S']
        else:
            ans+=['B']
            ans+=['M']*(len(item)-2)
            ans+=['E']
    return ''.join(ans)

### 然后将分词的结果进行标号与embedding层之后的词向量进行相加

###   2.Ex-softword
'''
softword是一个分词然后将词的边界拼接进embedding层，而Ex-softword则是多种分词情况之后，
直接把该字符在上下文中所有能匹配上词汇信息都加入进来 如下所示每个字符会遍历词表，拿到所有包含这个字且满足前后文本的所有分词tag
通俗来讲就是在这句话中当前词可以充当什么成分，可以充当B？可以充当M?可以充当E?
'''
###   3.Softlexicon

'''
default = {'B' : set(), 'M' : set(), 'E' : set(), 'S' :set()}
soft_lexicon = [deepcopy(default) for i in range(len(sentence))]

for i in range(len(sentence)):
    for j in range(i, min(i+MaxWordLen, len(sentence))):
        word = sentence[i:(j + 1)]
        if word in Vocab2IDX:
            if j-i==0:
                soft_lexicon[i]['S'].add(word)
            elif j-i==1:
                soft_lexicon[i]['B'].add(word)
                soft_lexicon[i+1]['E'].add(word)
            else:
                soft_lexicon[i]['B'].add(word)
                soft_lexicon[j]['E'].add(word)
                for k in range(i+1, j):
                    soft_lexicon[k]['M'].add(word)
    for key, val in soft_lexicon[i].items():
        if not val:
            soft_lexicon[i][key].add(NONE_TOKEN)
'''

