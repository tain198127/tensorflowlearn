import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    实现缩放点积注意力机制
    :param query: 查询向量
    :param key: 键向量
    :param value: 值向量
    :param mask: 注意力掩码（可选）
    :return: 注意力加权后的值向量
    """
    matmul_qk = np.matmul(query, key.T)
    d_k = query.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = np.exp(scaled_attention_logits)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, value)
    return output, attention_weights

def multihead_attention(query, key, value, num_heads):
    """
    实现多头注意力机制
    :param query: 查询向量
    :param key: 键向量
    :param value: 值向量
    :param num_heads: 头的数量
    :return: 多头注意力加权后的值向量
    """
    d_model = query.shape[-1]
    assert d_model % num_heads == 0
    head_dim = d_model // num_heads

    # 将输入向量线性映射为不同头的查询、键、值
    query = np.reshape(query, (-1, num_heads, head_dim))
    key = np.reshape(key, (-1, num_heads, head_dim))
    value = np.reshape(value, (-1, num_heads, head_dim))

    # 在每个头上计算注意力
    attention_output, _ = scaled_dot_product_attention(query, key, value)

    # 合并多头的输出
    attention_output = np.reshape(attention_output, (-1, d_model))
    return attention_output

# 用一个简单的例子来解释：
# 假设你在玩积木，每个积木都有不同的形状（query）、颜色（key）和大小（value）。
# 你想要按照形状、颜色和大小来组织这些积木。
# 首先，你将每种属性映射到不同的头上（多头注意力机制），
# 然后在每个头上计算注意力，最后将多头的注意力加权结果合并起来。

# 请注意，这只是 Transformer 中的一小部分核心算法，实际的 Transformer 模型更为复杂。
# 创建示例数据
query = np.array([[0.5, 0.3, 0.8]])
key = np.array([[0.1, 0.7, 0.2]])
value = np.array([[0.2, 0.6, 0.3]])
output, attention_weights = scaled_dot_product_attention(query, key, value)
print("Scaled Dot-Product Attention Output:")
print(output)
print("Attention Weights:")
print(attention_weights)