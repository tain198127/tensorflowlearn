import torch
import torch.nn as nn
import torchtext
import random
print(torchtext.__version__)
from torchtext import data
from torchtext import datasets
import torch
flag = torch.backends.mps.is_available()
if flag:
    print("mps可用")
else:
    print("CUDA不可用")
# import torchvision
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
tokenizer = word_tokenize

# torchtext
SEED=1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
TEXT = data.Field(tokenize=tokenizer,include_lengths=True)
LABEL=data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT,LABEL)
print('IMDB分类完毕')
train_data,valid_data = train_data.split(random_state = random.seed(SEED))
MAX_VOCAB_SIZE=25000
TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.300d",
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
print("build ok")


BATCH_SIZE=64
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
train_iter,valid_iter,test_iter = data.BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device
)
print('build iter 结束')

class RNN(torch.nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_lyaers,
                 bidirectional, dropout,pad_idx):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim=embedding_dim,padding_idx=pad_idx)
        self.rnn=nn.LSTM(embedding_dim,hidden_size=hidden_dim,num_layers=n_lyaers,
                         bidirectional=bidirectional,dropout=dropout)
        self.fc=nn.Linear(hidden_dim*2,output_dim)
        self.dropout=nn.Dropout(dropout)
    def fowward(self,text,text_lengths):
        embedded=self.embedding(text)
        packed_embedded=nn.utils.rnn.pad_packed_sequence(embedded,text_lengths)
        packed_output,(hidden,cell)=   self.rnn(packed_embedded)
        output,output_lengths=nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat(hidden[-2,:,:],hidden[-1,:,:],dim=1))
        return self.fc(hidden)

INPUT_DIM=len(TEXT.vocab)
EMBEDDING_DIM=300
HIDDEN_DIM=256
OUTPUT_DIM=1
N_LAYERS=2
BIDIRECTIONAL=True
DROPOUT=0.2
PAD_IDX=TEXT.vocab.stoi(TEXT.pad_token)

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,DROPOUT,PAD_IDX)
pretrained_embeddings=TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX=TEXT.vocab.stoi(TEXT.unk_token)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX]=torch.zeros(EMBEDDING_DIM)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion=nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)
