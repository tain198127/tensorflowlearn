import torch
import torch.nn as nn

t1 = torch.tensor([[1.0, 1.2], [2.1, 2.2]], requires_grad=True)
result = t1.pow(4).sum()
result.backward()
# print(t1.grad)
# print(t1)
t2 = torch.tensor([1, 2, 3])
t3 = torch.tensor([6, 7, 8, 9, 10])
# print(t2)
# print(t3)
t4 = torch.einsum("x,y->xy", t2, t3)
t0 = torch.tensor([1, 2, 4])
# print(t4)


x = torch.rand([8, 100, 10]).detach()
y = torch.rand(8)
y = (y > 0.5).int()
print(y)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(1000, 50)
        self.second_layer = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = nn.functional.relu(self.first_layer(x))
        x = self.second_layer(x)
        return x


mlp = MLP()
output = mlp(x)
print(output)


class Embedding(torch.nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(8, 100)

    def forward(self, input):
        return self.embedding(input)

emb = Embedding()
emb_input = torch.tensor([[1,2,3,4],[7,5,4,3],[4,4,3,2]])
emb_output = emb(emb_input)
print(emb_output)


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        #这里的10表示的是embedding的dim
        #15是隐藏层的dim
        # num_layers表示是几层的意思
        # bidirectional表示的是单向还是双向，true表示双向
        self.sltm = nn.LSTM(10,15,num_layers=2,bidirectional=True,dropout=0.1)
    def forward(self,x):
        output,(hiden,cell) = self.sltm(x)
        return output, hiden,cell
# permute表示换维度，相当于，例如原来的维度是[8,100,10]换维后，变成[100,8,0]
permute_x = x.permute([1,0,2])
print("permute:{}",permute_x.shape)

lstm = LSTM()
output_lstm1,output_hidden,output_cell = lstm(permute_x)
# 这里的shape变成torch.Size([100, 8, 30])，30是因为lstm里面，
# 我用的是双向的num_layer2=2，并且我用的隐藏层是15，双向的。正向反向15*2，因此维30
print('lstm output1 is',output_lstm1.shape)
# 这里的输出是torch.Size([4, 8, 15])，4表示num_layer的数量，虽然num_layer的数量是2由于是双向的，因此是4
#15就是hidden_dim的意思
print('lstm hidden output is',output_hidden.shape)
print('lstm cell output is ',output_cell.shape)

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(100,50,2)
    def forward(self,x):
        return self.conv(x)
conv = Conv()
output = conv(x)
print('conv1d is',output.shape)