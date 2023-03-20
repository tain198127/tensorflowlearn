import transformers
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf

transformers.__version__
print(transformers.__version__)


# download bert-base-uncased model
model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence = 'I love Beijing'

tokens = tokenizer.tokenize(sentence)
print("tokes")
print(tokens)

tokens = ['[CLS]'] + tokens + ['[SEP]']
print("tokes add cls and sep")
print(tokens)

tokens = tokens + ['[PAD]'] * 2
print("tokes add pad")
print(tokens)


attention_mask = [ 1 if t != '[PAD]' else 0 for t in tokens]
print("tokes add attention")
print(attention_mask)

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(token_ids)
#
# output = model(token_ids, attention_mask = attention_mask)
# print(output[0].shape, output[1].shape)