import numpy as np
import torch
from data_process.dataloader import DataLoader
from data_process.datasets import DataSets

print('程序启动')
datas = DataSets(path='E:/Downloads/Compressed/flower_photos')
dl = DataLoader(datas, 6, datas.collate_fn)

x_train, y_train = None, None
for i in dl:
    x_train, y_train = i

n, m = x_train.shape
nh = 40000
w1 = torch.randn(m, nh, requires_grad=True)
b1 = torch.zeros(nh, requires_grad=True)
w2 = torch.randn(nh, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)


def forward(x, y):
    z1 = linear(x, w1, b1)
    a1 = torch.relu(z1)
    z2 = linear(a1, w2, b2)
    return loss(z2, y)

def linear(x, w, b):
    return x @ w + b

def loss(pre, lab):
    return (pre[:0] - lab).pow(2).mean()

ans = forward(x_train, y_train)

ans.backward()
print(1)

# def lin_grad(inp, out, w, b):
#     inp.g = out.g * w.t()
#     w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
#     b.g = out.g.sum(0)

# z1 = x * w1 + b1
z1 = linear(x_train, w1, b1)
# a1 = relu(z1)
a1 = torch.relu(z1)
# z2 = a * w2 + b2
z2 = linear(a1, w2, b2)
# loss = (l2-lab)**2.mean()
# loss/z2 = 2 * diff / shape
diff = z2[:, 0] - y_train
z2.g = 2 * diff[:, None] / x_train.shape[0]
# loss/w2 = loss/z2 * l2/w2 = z2.g * a1
w2.g = (a1.unsqueeze(-1)*z2.g.unsqueeze(1)).sum(0)
# loss/b2 = loss/z2 * l2/b2 = z2.g * 1
b2.g = z2.g.sum(0)
# loss/a1 = loss/z2 * z2/a1 = z2.g * w2
a1.g = z2.g * w2.t()
# loss/z1 = loss/a1 * a1/z1
z1.g = (z1 > 0).float() * a1.g
# loss/w1 = loss/z1 * z1/w1 = z1.g * x
w1.g = (x_train.unsqueeze(-1) * z1.g.unsqueeze(1)).sum(0)
# loss/b1 = loss/z1 * z1/b1 =  z1.g
b1.g = z1.g.sum(0)




# loss = forward(x_train, y_train)
# loss.backward()
# print(1)







