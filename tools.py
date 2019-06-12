import os
import jieba
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import time


# 传入文本路径读取成string
def read_text_file(filename, ec='gb2312'):  # 系统默认gb2312
    with open(filename, "r", encoding=ec) as f:
        text = f.read()
    return text


# 传入停用词文本，按行读取，生成停用词集合
def build_stop_word_list(stop_word_str):
    stop_words = set()
    str_split = stop_word_str.split('\n')
    for line in str_split:
        stop_words.add(line.strip())

    stop_words.add('\n')
    stop_words.add('\t')
    stop_words.add(' ')

    return stop_words


# 读取目录所有文件
def read_dir(dir_path):
    file_list = []
    file_name_list = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        file_list.append(file_path)
        file_name_list.append(file.replace(".txt", ""))
    return file_list, file_name_list


# 利用结巴分词进行中文分词，去除停用词，建立词语集合
def build_word_set(text, stop_words):
    words = ' '.join(jieba.cut(text)).split(' ')
    word_set = set()
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        if word not in stop_words:
            word_set.add(word)
    return word_set


# 合并所有集合生成df轴
def get_axis(set_list):
    axis_set = set()
    for item in set_list:
        axis_set = axis_set.union(item)
    return axis_set


# 加载词向量语料库
def load_model(corpus):
    wv_from_text = KeyedVectors.load_word2vec_format(corpus, binary=False)
    return wv_from_text


# 画热点图
def draw(df, axis):
    df = df + df.values.T  # 加上转置
    df.index = axis
    df.columns = axis

    sns.set(font='STSong')  # 解决中文字体显示
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax, cmap="vlag")

    plt.show()


# 计算词语语意相似度(原始)
def comp_sim_ori(set1, set2, wv):
    N1 = len(set1)
    N2 = len(set2)

    df = pd.DataFrame(np.zeros([N1, N2]))

    for i, w1 in enumerate(set1):
        for j, w2 in enumerate(set2):
            try:
                w_sim = wv.similarity(w1, w2)  # 计算两个词相似度
            except:
                w_sim = 0
            df.iloc[i, j] = w_sim

    # 行列都取最大值，然后合起来求平均
    v0 = df.dropna().max(axis=0)
    v1 = df.dropna().max(axis=1)
    avg_sim = ((v0.sum() + v1.dropna().sum()) /
               (len(v0.dropna()) + len(v1.dropna())))

    return avg_sim


def comp_sim_np(set1, set2, wv):
    N1 = len(set1)
    N2 = len(set2)

    df = np.mat(np.zeros((N1, N2)))
    n = 0
    for i, w1 in enumerate(set1):
        for j, w2 in enumerate(set2):
            try:
                w_sim = wv.similarity(w1, w2)  # 计算两个词相似度
                n = n + 1
            except:
                w_sim = 0
            df[i, j] = w_sim
    print("计算次数：", n, N1, N2)
    # 行列都取最大值，然后合起来求平均
    v0 = np.max(df, axis=0).sum()
    v1 = np.max(df, axis=1).sum()
    avg_sim = ((v0 + v1) / (N1 + N2))

    return avg_sim


# 计算词语语意相似度(优化)
def comp_sim_mark1(set1, set2, wv, df_mark):
    N1 = len(set1)
    N2 = len(set2)
    n = 0
    # df = pd.DataFrame(np.zeros([N1, N2]))
    df = np.mat(np.zeros((N1, N2)))

    for i, w1 in enumerate(set1):
        for j, w2 in enumerate(set2):
            # if w1 == w2:
            #     w_sim = 1
            # else:
            #     w_sim = df_mark.loc[w1, w2]
            #
            # if w_sim == 0:
            try:
                t = time.time()
                w_sim = wv.similarity(w1, w2)  # 计算两个词相似度
                n = n + time.time() - t
                # df_mark.loc[w1, w2] = w_sim
                # df_mark.loc[w2, w1] = w_sim

            except:
                # 比较出错，标志-1
                w_sim = 0
                # df_mark.loc[w1, w2] = -1
                # df_mark.loc[w2, w1] = -1

            # if w_sim == -1:
            # w_sim = 0
        df[i, j] = w_sim
    print("计算时间", n)
    # 行列都取最大值，然后合起来求平均
    v0 = np.max(df, axis=0).sum()
    v1 = np.max(df, axis=1).sum()
    avg_sim = ((v0.sum() + v1.sum()) / (N1 + N2))
    #
    # v0 = df.dropna().max(axis=0)
    # v1 = df.dropna().max(axis=1)
    # avg_sim = ((v0.sum() + v1.dropna().sum()) / (len(v0.dropna()) + len(v1.dropna())))
    return avg_sim


def compSim_mark(set1, set2, wv, mark, record, wordlist):
    N1 = len(set1)
    N2 = len(set2)
    fSim = 0

    df = np.mat(np.zeros([N1, N2]))

    w1_pos = 0
    i = 0
    j = 0
    f = 0
    n = 0
    for w1 in set1:
        w1_pos = 0
        judge = 0
        j = 0
        for w2 in set2:
            if w1 == w2:
                f = 1
            else:
                if judge == 0:
                    if w1 in record:
                        w1_pos = wordlist.index(w1)

                    else:
                        record.add(w1)
                        wordlist.append(w1)
                        w1_pos = len(wordlist) - 1

                    judge = 1

                if w2 in record:
                    w2_pos = wordlist.index(w2)

                    f = mark[w1_pos, w2_pos]
                    if f == 0:  # 没比较过
                        try:

                            f = wv.similarity(w1, w2)

                        except:
                            f = -1

                        mark[w1_pos, w2_pos] = f
                        mark[w2_pos, w1_pos] = f

                else:
                    record.add(w2)
                    wordlist.append(w2)
                    w2_pos = len(wordlist) - 1
                    try:

                        f = wv.similarity(w1, w2)

                    except:
                        f = -1

                    mark[w1_pos, w2_pos] = f
                    mark[w2_pos, w1_pos] = f

            df[i, j] = f
            j = j + 1

        i = i + 1

    v0 = np.max(df, axis=0).sum()
    v1 = np.max(df, axis=1).sum()
    fSim = ((v0 + v1) / (N1 + N2))

    return fSim
