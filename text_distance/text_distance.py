import io
import sys
import Levenshtein
import time
import numpy as np
from nltk.metrics import edit_distance

tt = time.time()

count = 10
r = np.mat(np.ones((count, count)))
f_list = []

# 载入文件
for i in range(count):
    with open('./data/%s.txt' % (i + 1)) as f1:
        text_list = f1.read().split(" ")
        f_list.append(set(text_list))


# 计算两个文件的文本距离
def text_distant(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    sum = 0
    for i in set1:
        for j in set2:
            a = Levenshtein.distance(i, j)
            #a = edit_distance(i, j)
            sum = sum + a
    dist = sum / (len1 * len2)
    return dist


# 生成矩阵
for i, item in enumerate(f_list):
    set1 = item
    for j in range(i + 1, len(f_list)):
        set2 = f_list[j]
        distance = text_distant(set1, set2)
        r[i, j] = distance

# 改变标准输出的默认编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
print("10个文件对比时间：", time.time() - tt)
