# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/5/14 13:57
coding with comment！！！
"""

def get_len(tetx):
    """
    Length filtering
    :param tetx:
    :return:
    """
    if len(tetx) <= 7:
        return False
    return True

def is_chinese(text):
    """
    Check if the given text contains Chinese characters.
    """
    # Unicode range for Chinese characters
    import re
    chinese_char_pattern = re.compile(
        r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\u2ceb0-\u2ebef]')

    if chinese_char_pattern.search(text):
        return True
    return False

def contains_alphanumeric(text):
    """
    Remove samples containing numbers and letters
    :param text:
    :return:
    """
    import re
    return not bool(re.search(r'[a-zA-Z0-9]', text))






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

    for line in lines:
        if not any(is_similar(line, unique_line, ratio_threshold) for unique_line in unique_lines):
            unique_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)


if __name__ == "__main__":
    data_path = r"C:\Users\NewNLPer\Desktop\school_rule _2.txt"

    save_path_del_ = r"C:\Users\NewNLPer\Desktop\school_rule _3.txt"
    fw = open(save_path_del_,"w",encoding="utf-8")
    save_path_remove_ = r"C:\Users\NewNLPer\Desktop\school_rule_4.txt"

    with open(data_path,"r",encoding='utf-8') as f:
        lines = f.readlines()
        for item in lines:
            if get_len(item) and contains_alphanumeric(item) and is_chinese(item):
                fw.write(item + "\n")


# # 使用示例
    ratio_threshold = 0.5  # 设定比例阈值，例如0.5表示最长公共子串占较短字符串的50%
    remove_similar_lines_based_on_lcs(save_path_del_, save_path_remove_, ratio_threshold)



