# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/2/8 17:34
coding with comment！！！
"""
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
import nltk.stem
# nltk.download('punkt')
# nltk.download('stopwords')
def train_pre(text):
    lower=text.lower()
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = lower.translate(remove)
    tokens = nltk.word_tokenize(without_punctuation)
    without_stopwords=[]
    for w in tokens:
        if w not in stopwords.words('english') and not w.isdigit():
            without_stopwords.append(w)
    s = nltk.stem.SnowballStemmer('english')  # 参数是选择的语言
    cleaned_text = [s.stem(ws) for ws in without_stopwords]
    return ' '.join(cleaned_text)

# train_df1 = pd.read_csv(r"C:/Users/NewNLPer/Desktop/za/城堡竞赛/pre_traindata.csv")
train_df2=pd.read_csv(r"C:/Users/NewNLPer/Desktop/za/城堡竞赛/test.csv")
print(train_df2.columns)
train_df2.columns = ['Label','txt']
print(train_df2.columns)

for i in range(25000):
    # train_df1['txt'][i]=train_pre(train_df1['txt'][i])
    train_df2['txt'][i] = train_pre(train_df2['txt'][i])
    print(i)
# train_df1[["txt",'Label']].to_csv('C:/Users/NewNLPer/Desktop/za/城堡竞赛/pre_traindata.csv',index=False)
train_df2[['txt','Label']].to_csv('C:/Users/NewNLPer/Desktop/za/城堡竞赛/pre_testdata.csv',index=False)



