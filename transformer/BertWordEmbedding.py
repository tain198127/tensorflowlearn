from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

# 加载BERT模型和tokenizer（这里使用BERT-base模型）
# model_name = 'bert-base-uncased'
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入文本
text = "我非常喜欢这种算法，你感觉如何？"

# 使用tokenizer将文本转换为BERT输入格式
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
# 调用BERT模型获得整个文本的词嵌入
with torch.no_grad():
    outputs = model(**tokens)

# 获取整个文本的词嵌入
embeddings = outputs.last_hidden_state[0]  # 使用[0]获取第一个样本的词嵌入

# 获取每个子词的向量
subword_tokens = tokens['input_ids'][0]
subword_embeddings = []

for i, token_id in enumerate(subword_tokens):
    # 跳过特殊标记（如[CLS]和[SEP]）
    if token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
        subword_embedding = embeddings[i]
        subword_embeddings.append(subword_embedding)
        subword = tokenizer.convert_ids_to_tokens(token_id)
        print("子词：", subword)
        print("子词的词嵌入：", subword_embedding)
# similarity = cosine_similarity(word1_emb.unsqueeze(0), word2_emb.unsqueeze(0))
# print("余弦相似度：", similarity.item())
