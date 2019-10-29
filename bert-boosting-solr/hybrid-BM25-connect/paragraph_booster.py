import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torch_multi_head_attention import MultiHeadAttention
import helper_functions


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        attention = F.softmax(scores, dim=1)
        return attention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        y = ScaledDotProductAttention()(q, k, v)
        return y

    @staticmethod

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
class ParaBoost(nn.Module):
  def __init__(self, embedding_size):
    super(ParaBoost, self).__init__()
    self.embedding_size = embedding_size
    self.NN = nn.Linear(2,1)
    self.multi_head_attention = MultiHeadAttention(in_features=(embedding_size * 9), head_num=12)

  def forward(self, word_embs, para_embs):
    num_words = word_embs.size()[0]
    word_embs = word_embs.unsqueeze(0)
    para_embs = para_embs.unsqueeze(0)
    para_embs = para_embs.unsqueeze(0)
    new_embs = self.multi_head_attention(word_embs, para_embs, para_embs)
    new_embs = new_embs.view(12,-1)
    new_embs = new_embs.sum(0)
    new_embs = torch.cat((new_embs.unsqueeze(0), columns2.unsqueeze(0)),0)
    new_embs = self.NN(new_embs.transpose(-1,-2))
    new_embs = new_embs.transpose(-1,-2)
    return new_embs
