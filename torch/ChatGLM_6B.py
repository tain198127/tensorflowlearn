import faulthandler
faulthandler.enable()
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/Users/baodan/develop/chatglm_6b_4int", trust_remote_code=True)
print('lod tokenizer')
model = AutoModel.from_pretrained("/Users/baodan/develop/chatglm_6b_4int", trust_remote_code=True).float()
print('load model')
response, history = model.chat(tokenizer, "你好", history=[])
while True:
    message = input("请输入你的问题")
    if message == 'stop':
        break
    elif message == 'clear':
        history = []
        continue
    response, history = model.chat(tokenizer, message, history=history)
    print(response)
