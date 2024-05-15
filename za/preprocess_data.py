# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/14 13:57
coding with comment！！！
"""
from tqdm import tqdm
def remove_similar_lines_based_on_lcs(input_file, output_file, ratio_threshold):
    def longest_common_substring(s1, s2):
        """
        Find the longest common substring between two strings.
        """
        m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
        longest, x_longest = 0, 0
        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0
        return s1[x_longest - longest: x_longest]

    def is_similar(line1, line2, ratio_threshold):
        """
        Check if the longest common substring between two lines is greater than the ratio threshold.
        """
        lcs = longest_common_substring(line1, line2)
        min_length = min(len(line1), len(line2))
        return len(lcs) / min_length > ratio_threshold

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    unique_lines = []

    for line in tqdm(lines):
        if not any(is_similar(line, unique_line, ratio_threshold) for unique_line in unique_lines):
            unique_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)


# def get_split_rule(data_path):
#     save_list = []
#     with open(data_path,"r",encoding="utf-8") as f:
#         lines = f.readlines()
#         start = 0
#         while start < len(lines):
#             if "条" in lines[start] and "第" in lines[start]:
#                 end = start + 1
#                 while end < len(lines):
#                     if "条" in lines[end] and "第" in lines[end] or ( "章" in lines[end]):
#                         save_list.append(''.join(lines[start:end]))
#                         start = end
#                         break
#                     else:
#                         end += 1
#                 else:
#                     break
#             else:
#                 start += 1
#     for i in range(len(save_list)):
#         save_list[i] = save_list[i].replace("\n","")
#     return save_list

if __name__ == "__main__":
    # data_path = r"C:\Users\NewNLPer\Desktop\school_rule.txt"
    save_path = r"C:\Users\NewNLPer\Desktop\school_rule_pre_1.txt"
    save_path_preLSC = r"C:\Users\NewNLPer\Desktop\school_rule_pre_2.txt"
    # fw = open(save_path,"w",encoding="utf-8")
    # precess_ = get_split_rule(data_path)
    # for item in precess_:
    #     print(item)
    #     print('======================================================')
    #     fw.write(item + "\n")

    # ratio_threshold = 0.5  # 设定比例阈值，例如0.5表示最长公共子串占较短字符串的50%
    # remove_similar_lines_based_on_lcs(save_path, save_path_preLSC, ratio_threshold)
    with open(save_path_preLSC,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            print('======================')