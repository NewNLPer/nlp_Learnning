# -*- coding:utf-8 -*-
# @Time      :2022/8/18 18:14
# @Author    :Riemanner
# Write code with comments !!!
import collections
def maxEqualFreq(nums: list[int]) -> int:
    # 四种情况：（1）12345 11111；（2）1122334；（3）112233444。
    # 设定两个哈希表 很拗口！
    # 第一个用于存储 数字x出现的次数
    num_freq = collections.defaultdict(int)
    # 第二个用于存储 出现次数为f的个数
    freq_freq = collections.defaultdict(int)
    # 设定出现的最大次数 以及 数组中涉及的数字x种类
    maxFreq, species = 0, 0
    # 最特殊的结果
    res = 0
    # 开始循环
    for i, num in enumerate(nums):
        #  只要出现新的数字 那么种类便加一
        if num_freq[num] == 0: species += 1
        # 对应数字频率加一
        num_freq[num] += 1
        # 注意
        # 最大频率
        maxFreq = max(maxFreq, num_freq[num])
        # 频率为 num_freq[num]的个数+1
        freq_freq[num_freq[num]] += 1
        # 频率为num_freq[num]-1的个数-1
        freq_freq[(num_freq[num] - 1)] -= 1

        # 123456 111111
        if species == i + 1 or species == 1:
            res = i + 1
        # 1122334
        if freq_freq[maxFreq] == species - 1 and freq_freq[1] == 1:
            res = i + 1
        # 112233444
        if freq_freq[maxFreq] == 1 and freq_freq[maxFreq - 1] == species - 1:
            res = i + 1
    return res
print(maxEqualFreq([1,1,1,2,2,2,3,3,3]))

