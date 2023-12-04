"""
@author: NewNLPer
@time: 2022/12/8 9:49
coding with comment！！！
"""
## 导入本章所需要的模块
import pandas as pd
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname = "/Library/Fonts/华文细黑.ttf")## 输出图显示中文
import jieba

###最初始的文本数据
train_df = pd.read_csv("D:/charm/pytorch数据/程序/programs/data/chap7/cnews/cnews.train.txt",sep="\t",
                       header=None,names = ["label","text"])
val_df = pd.read_csv("D:/charm/pytorch数据/程序/programs/data/chap7/cnews/cnews.val.txt",sep="\t",
                       header=None,names = ["label","text"])
test_df = pd.read_csv("D:/charm/pytorch数据/程序/programs/data/chap7/cnews/cnews.test.txt",sep="\t",
                       header=None,names = ["label","text"])
###中文停用词，方便后续分词处理
stop_words = pd.read_csv("D:/charm/pytorch数据/程序/programs/data/chap7/cnews/中文停用词库.txt",sep="\t",
                         header=None,names = ["text"])
def chinese_pre(text_data):
    # text_data = text_data.lower()    ## 字母转化为小写
    # text_data = re.sub("\d+", "", text_data)   ## 去除数字，正则表达是的强大之处
    text_data = list(jieba.cut(text_data,cut_all=False))      ## 分词,使用精确模式(值得注意的是，jieba自动采用HMM来进行计算最大概率，从而实现分词)
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]     ## 去停用词和多余空格
    text_data = " ".join(text_data) ## 处理后的词语使用空格连接为字符串
    return text_data

###构建一个新的标题’cutword‘将对text的分词处理进行记录，chinese_pre函数要记住，
train_df["cutword"] = train_df.text.apply(chinese_pre)
val_df["cutword"] = val_df.text.apply(chinese_pre)
test_df["cutword"] = test_df.text.apply(chinese_pre)

###将第一个标题进行修改。
labelMap = {"体育": 0,"娱乐": 1,"家居": 2,"房产": 3,"教育": 4,
            "时尚": 5,"时政": 6,"游戏": 7,"科技": 8,"财经": 9}
train_df["labelcode"] =train_df["label"].map(labelMap)
val_df["labelcode"] =val_df["label"].map(labelMap)
test_df["labelcode"] =test_df["label"].map(labelMap)

###数据处理完成，将所以需要的进行保存。


train_df[["labelcode","cutword"]].to_csv("data/chap7/cnews_train2.csv",index=False)
val_df[["labelcode","cutword"]].to_csv("data/chap7/cnews_val2.csv",index=False)
test_df[["labelcode","cutword"]].to_csv("data/chap7/cnews_test2.csv",index=False)

