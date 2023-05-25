#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
#写在最前
#在mac上运行，先要
# curl -O https://mac.r-project.org/openmp/openmp-14.0.6-darwin20-Release.tar.gz
# sudo tar fvxz openmp-14.0.6-darwin20-Release.tar.gz -C /
# 加载openmp才可以
# 然后要安装 pip install -r requirements.txt
#如果想用mac m1 max的mps 需要如下步骤：
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
# sh Miniconda3-latest-MacOSX-arm64.sh
# conda install pytorch torchvision torchaudio -c pytorch-nightly

#
#
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
