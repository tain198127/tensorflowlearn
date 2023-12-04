import gensim
import numpy as np
import time
import datetime
found = 0

def compute_ngrams(word, min_n, max_n):
    extended_word = word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return list(set(ngrams))


def word_vector(word, wv_from_text, min_n=1, max_n=3):
    # 确认词向量维度
    word_size = wv_from_text.vectors.shape[0]
    # 计算word的ngrams词组
    ngrams = compute_ngrams(word, min_n=min_n, max_n=max_n)
    # 如果在词典之中，直接返回词向量
    if word in wv_from_text.index_to_key:
        global found
        found += 1
        return wv_from_text[word]
    else:
        # 不在词典的情况下，计算与词相近的词向量
        word_vec = np.zeros(word_size, dtype=np.float32)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams if len(ng) > 1]
        # 先只接受2个单词长度以上的词向量
        for ngram in ngrams_more:
            if ngram in wv_from_text.index_to_key:
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
                # print(ngram)
        # 如果，没有匹配到，那么最后是考虑单个词向量
        if ngrams_found == 0:
            for ngram in ngrams_single:
                if ngram in wv_from_text.index_to_key:
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
        if word_vec.any():  # 只要有一个不为0
            return word_vec / max(1, ngrams_found)
        else:
            print('all ngrams for word %s absent from model' % word)
            return 0


if __name__ == '__main__':
    print("开始载入文件...")
    print("Now：", datetime.datetime.now())
    t1 = time.time()
    # wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(
    # '/Users/baodan/develop/ai/Tencent_AILab_ChineseEmbedding.txt', limit=4000000,
    #                                                                binary=False)
    # wv_from_text.init_sims(replace=True)
    # wv_from_text.save(r"E:\ModelFolder\400million\ChineseEmbedding.bin")
    wv_from_text = gensim.models.KeyedVectors.load(
        '/Users/baodan/develop/ai/Tencent_AILab_ChineseEmbedding.bin',mmap='r')
    print("文件载入完毕")


    # print(wv_from_text.index2word)
    print("文件载入耗费时间：", (time.time() - t1) / 60, "minutes")
    # result_list = open_file("keyword.txt")
    print("获取关键词列表")
    input_text = "苹果，原装，手机"
    result_list = input_text.split("，")
    words_length = len(result_list)
    print(result_list)

    for keyword in result_list:
        vec = word_vector(keyword, wv_from_text, min_n=1, max_n=3)  # 词向量获取
        if vec is 0:
            continue
        # print("获取的词向量：", vec)
        similar_word = wv_from_text.most_similar(positive=[vec], topn=15)  # 相似词查找
        result_word = [x[0] for x in similar_word]
        print(result_word)
    print("词库覆盖比例：", found, "/", words_length)
    print("词库覆盖百分比：", 100 * found / words_length, "%")
    print("整个推荐过程耗费时间：", (time.time() - t1) / 60, "minutes")