from tools import *
import time
import csv


def run(wv, doc_set_list, book_name, file_name="pic"):
    # 初始化标志矩阵
    axis_set = get_axis(doc_set_list)
    axis_len = len(axis_set)
    mark = np.mat(np.zeros(([axis_len, axis_len])))

    record = set()
    wordlist = []

    # 计算文档相似度，存入df
    L = len(doc_set_list)
    df = pd.DataFrame(np.zeros([L, L]))

    start = time.time()
    for i in range(L - 1):
        set1 = doc_set_list[i]
        for j in range(i + 1, L):
            set2 = doc_set_list[j]
            # df.iloc[i, j] = comp_sim(set1, set2, wv, df_mark)  # 有标志矩阵
            # df.iloc[i, j] = comp_sim_np(set1, set2, wv)  # 无有标志矩阵
            df.iloc[i, j] = compSim_mark(set1, set2, wv, mark, record, wordlist)  # 有标志矩阵
            # df.iloc[i, j] = comp_sim0(set1, set2, wv)  # 无有标志矩阵
            # print(i, j, "次 对比结果：", df.iloc[i, j], "对比时间：", time.time() - start)
    print(file_name + "结果时间：", time.time() - start)
    draw(df, book_name, file_name)
    return df


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

    with open('book_info.csv', 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    book_name = []
    tag_set_list = []
    brief_set_list = []
    cat_set_list = []

    tag = []
    brief = []
    cat = []

    n = 0
    for i in rows:
        book_name.append(i[0])
        tag = i[1]
        brief = i[2]

        cat = i[3]
        if cat == "":
            cat = i[1]

        tag_set_list.append(build_word_set(tag, stop_words))
        brief_set_list.append(build_word_set(brief, stop_words))
        cat_set_list.append(build_word_set(cat, stop_words))

        n = n + 1
        if n == 20:
            break

    tag_df = run(wv, tag_set_list, book_name, "tag")
    brief_df = run(wv, brief_set_list, book_name, "brief")
    cat_df = run(wv, cat_set_list, book_name, "cat")

    df = tag_df * 0.5 + brief_df * 0.2 + cat_df * 0.3
    pd.DataFrame(df).to_csv('res.csv')
    draw(df, book_name, "res")
