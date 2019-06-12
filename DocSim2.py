from tools import *
import time
import csv

if __name__ == "__main__":
    # （1）利用jieba分词，对多个txt文件，利用一个停用词文件后分词，返回多个 string set
    # （2）把多个set，利用 Gensim比较各自文件的词的语义相似性
    # （3）设计算法，利用词的相似性比较文件的相似性
    # （4）保存数据文件
    # （5）画热点图

    # 读取停用词文件生成停用词集合
    stop_word_file = r'./停用词_中文和符号1960.txt'
    encoding = 'UTF-8'
    stop_word_str = read_text_file(stop_word_file, encoding)

    stop_words = build_stop_word_list(stop_word_str)

    # 加载语料库
    corpus = r"./corpus/100000-small.txt"  # 小语料库
    wv = load_model(corpus)

    with open('comment.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    book_name = []
    doc_set_list = []

    n = 0
    for i in rows:
        book_name.append(i[0])
        doc_set_list.append(build_word_set(i[1], stop_words))
        n = n + 1
        if n==3:
            break

    # 初始化标志矩阵
    axis_set = get_axis(doc_set_list)
    axis_len = len(axis_set)
    mark = np.mat(np.zeros(([axis_len + 1, axis_len + 1])))

    record = set()
    wordlist = []

    # 计算文档相似度，存入df
    L = len(doc_set_list)
    df = pd.DataFrame(np.zeros([L, L]))

    start = time.time()
    # 双层循环，低效，需要改进
    for i in range(L - 1):
        set1 = doc_set_list[i]
        for j in range(i + 1, L):
            set2 = doc_set_list[j]
            # df.iloc[i, j] = comp_sim(set1, set2, wv, df_mark)  # 有标志矩阵
            # df.iloc[i, j] = comp_sim_np(set1, set2, wv)  # 无有标志矩阵
            df.iloc[i, j] = compSim_mark(set1, set2, wv, mark, record, wordlist)  # 有标志矩阵
            # df.iloc[i, j] = comp_sim0(set1, set2, wv)  # 无有标志矩阵
            print(i, j, "次 对比结果：", df.iloc[i, j], "对比时间：", time.time() - start)
    print(df)
    print("结果时间：", time.time() - start)
