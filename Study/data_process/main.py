import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as f
import torch.nn as nn
# input (2, 4) -> (2, 4, 512), 句子的长度，vocab,词表的大小


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.init_weight()

    def init_weight(self):
        for p in self.emb.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
# tf 他的每一层维度都是相同的


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # (batch, len, dmodel) (5000, 1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = 1 / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        # div_term = torch.exp(torch.arange(0, d_model, 2) *
        #                      -(math.log(10000.0) /
        #                        d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不参与训练

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequence_mask(size):
    atten_shape = (1, size, size)
    _subsequence_mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    return torch.from_numpy(_subsequence_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = f.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn




if __name__ == '__main__':
    s = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    t = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    embedding = Embedding(512, 11)
    posi = PositionEncoding(512)
    x = embedding(s)
    x = posi(x)
    m = subsequence_mask(x.shape[1])
    x = attention(x, x, x)
    print(x)

    # 句子的长度 len -》 我真的好好喜欢 7   vocab- 》10000000

