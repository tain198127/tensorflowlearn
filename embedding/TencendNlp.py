from gensim.models import KeyedVectors
import json
from collections import OrderedDict
from annoy import AnnoyIndex
import gensim

wv_from_text = KeyedVectors.load_word2vec_format('/Users/baodan/develop/ai/Tencent_AILab_ChineseEmbedding.txt', binary=False)
print("加载完毕\n")
wv_from_text.init_sims(replace=True)
wv_from_text.save("/Users/baodan/develop/ai/Tencent_AILab_ChineseEmbedding.bin")
print("转换为二进制\n")


