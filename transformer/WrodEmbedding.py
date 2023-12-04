from gensim.models import word2vec
import torch as pt
import numpy as np
# 准备训练数据（这里假设corpus是一个已经分好词的文本语料库）
corpus = [["我", "爱", "吃", "苹果"],
          ["苹果", "是", "水果"],
          ["我", "喜欢", "吃", "橙子"]]

# 训练Word2Vec模型
model = word2vec.Word2Vec(sentences=corpus,min_count=1, vector_size=100, window=5, sg=1)

# 获取单词的词嵌入向量
vector1 = model.wv['苹果']
print("苹果的词嵌入向量：", vector1)
vector2 = model.wv["水果"]
print("水果的词嵌入向量:",vector2)

dot_product = np.dot(vector1, vector2)  # 计算内积
norm1 = np.linalg.norm(vector1)  # 计算向量1的范数（模）
norm2 = np.linalg.norm(vector2)  # 计算向量2的范数（模）

cosine_similarity = dot_product / (norm1 * norm2)

print("余弦相似度：", cosine_similarity)

# pt1 = pt.from_numpy(vector1)
# pt2 = pt.from_numpy(vector2)
# dot_product = pt.dot(pt1,pt2)
# norm1 = pt.linalg.norm(pt1)
# norm2 = pt.linalg.norm(pt2)
# cosine_similarity = dot_product/(norm1* norm2)