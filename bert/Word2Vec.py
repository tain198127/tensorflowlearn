from nlpia.data.loaders import get_data
from gensim.models.keyedvectors import KeyedVectors

# 下载googlenews-vectors-negative300.bin.gz
# word_vectors = get_data('word2vec')

# 加载原始二进制格式的模型
word_vectors = KeyedVectors.load_word2vec_format('/Users/baodan/Downloads/googlenews-vectors-negative300.bin.gz',
                                                    binary=True)
# 从谷歌新闻语料库中加载最常用的 20 万个词
# limit 参数：减少加载到内存中的词的数量
# word_vectors_2w = KeyedVectors.load_word2vec_format('xxx\\googlenews-vectors-negative300.bin.gz',
#                                                     binary=True, limit=200000)

# 词向量
# 数组中的每个浮点数表示向量的一个维度
print(word_vectors['phone'])

# 查找最近相邻词
# gensim.KeyedVectors.most_similar()方法提供了对于给定词向量，查找最近的相
# 邻词的有效方法。关键字参数 positive 接受一个待求和的向量列表; negative 参数来做减法，以排除不相关的词项。
# 参数 topn 用于指定返回结果中相关词项的数量。
print(word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5))
print(word_vectors.most_similar(positive=['germany', 'france'], topn=5))

# doesnt_match():检测不相关的词项
print(word_vectors.doesnt_match("potatoes milk cake computer".split()))

# 词向量的计算
print( word_vectors.most_similar(positive=['king', 'woman'],  negative=['man'], topn=2))

# 词项余弦相似度的计算
print(word_vectors.similarity('princess', 'queen'))
