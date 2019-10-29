import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torch_multi_head_attention import MultiHeadAttention
import helper_functions

import functools 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




import numpy as np
import math
import os
import random
import sys
from io import open

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch import nn
from torch.nn import CrossEntropyLoss
sys.path.append('../../../../BERT-pytorch/pytorch-pretrained-BERT/')
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value):
        #print(query.size())
        #print(key.size())
        #print(value.size())
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        #print(scores.size())
        #print(scores)
        #print(scores.size())
        attention = F.softmax(scores, dim=-1)
        #print(attention)
        #attention = attention.sum(-2)
        #print(attention)
        #attention = attention.sum(-2)
        #print(attention)
        ''''
        print(value.size())
        test = attention[0][1]
        test2 = value.transpose(-1,-2)[0][2]
        print(test)
        print(test.size())
        print(test2)
        print(test2.size())
        print(test.matmul(test2))'''
        #helper_functions.torch_verbose_print(attention)
        #print(attention.matmul(value)[0][1][2])
        #attention = attention.matmul(value)
        #print(attention)
        #print("This does what you think it will")
        return attention.squeeze()

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
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
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        y = ScaledDotProductAttention()(q, k, v)
        return y
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

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
class BertBoosting(BertPreTrainedModel):
  def __init__(self, config, embedding_size=768):
    #Bert stuff
    super(BertBoosting, self).__init__(config)
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #self.classifier = nn.Linear(config.hidden_size, num_labels)
    self.apply(self.init_bert_weights)


    #my stuff
    num_heads = 12
    self.embedding_size = embedding_size
    self.probs_transform = nn.Linear(3, embedding_size)
    self.probs_layernorm = nn.LayerNorm(embedding_size)

    self.features_transform = nn.Linear(2, embedding_size)
    self.features_layernorm = nn.LayerNorm(embedding_size)
    #NOTE:conv layer same as linear layer, but requires view so use linear instead
    #self.query_conv = nn.Conv1d(1, 1, embedding_size, stride=embedding_size, padding=0)
    #self.query_conv.weight.data.fill_(1.)
    #self.query_conv.bias.data.fill_(0.)
    '''self.para_embs_layernorm = nn.LayerNorm(embedding_size*9)
    self.word_embs_layernorm = nn.LayerNorm(embedding_size*9)
    self.new_embs_layernorm = nn.LayerNorm(embedding_size*2)
    self.reduce_bert_layers_NN = nn.Linear(embedding_size*9, embedding_size)'''
    self.para_embs_layernorm = nn.LayerNorm(embedding_size)
    self.word_embs_layernorm = nn.LayerNorm(embedding_size)
    self.new_embs_layernorm = nn.LayerNorm(embedding_size*2)
    self.reduce_bert_layers_NN = nn.Linear(embedding_size, embedding_size)
    #self.reduce_bert_layers_NN = nn.Linear(embedding_size*9, 1)
    #self.NN = nn.Linear(embedding_size*2,1)
    self.BMtoHeads = nn.Linear(2,num_heads*2)
    self.NN = nn.Linear(num_heads*2,1)
    token_p_embedding = nn.Embedding(100, embedding_size)
    self.para_scores_NN = nn.Linear(1, embedding_size)
    self.query_idf_NN = nn.Linear(1, embedding_size)
    self.para_to_boost_NN = nn.Linear(embedding_size, 1)
    self.query_to_boost_NN = nn.Linear(embedding_size, 1)
    #self.NN.weight.data.fill_(10.)
    #self.NN.bias.data.fill_(0.)
    self.softmax = nn.Softmax()
    self.dropout_new_embs = nn.Dropout(.1)
    self.multi_head_attention = MultiHeadAttention(in_features=(embedding_size), head_num=num_heads)
  def map_to_new_tensor(self, tmap, old_tensor, device):
    new_tensor = [0.]*512
    old_tensor = old_tensor.float()
    new_tensor = torch.tensor(new_tensor).to(device).float()
    maximum = max(tmap)
    #print(tmap)
    tmap = torch.tensor(tmap).to(device)
    #print(tmap.size())
    #print(old_tensor.size())
    #print(new_tensor.size())
    #print(old_tensor)
    new_tensor.scatter_add_(0, tmap, old_tensor)
    #print(new_tensor)
    return torch.narrow(new_tensor, 0, 0, maximum+1)
  '''
  def compute_para_BMs(self, para_output2, p_to_qvocab, q_to_qvocab, query_idfs):
    para_output2 = [torch.split(p, 1, dim=0) for p in para_output2]
    vocab_BMs = [0.0]*max(p_to_qvocab)
    [[vocab_BMs[t]+=para_output[i][j] t for j,t in enumerate(p) for i,p] in enumerate(para_output2)]
  '''
  def convert_freq_to_embs(self, frequency_list, device, multiplier):
    frequency_list = np.round(np.array(frequency_list)*multiplier)
    #print(frequency_list)
    frequency_list = torch.clamp(torch.tensor(frequency_list).to(device),0,16383)
    #print(frequency_list.size())
    frequency_list = frequency_list.long()
    frequeny_list = frequency_list
    return frequency_list
  def get_layer_norm(self, vector):
    LN = torch.nn.LayerNorm(vector.size(), elementwise_affine=False)
    return LN(vector.float())
  def forward(self, device, q_to_qvocab=None, p_to_qvocab=None, denom2=None, para_scores=None, query_idfs=None, query_ids=None, \
             paragraph_ids = None, para_types=None, para_mask=None, query_types=None, query_mask=None, what_you_need=0, query_map=None, para_map = None ):
    k1=1.2
    b=.75
    if what_you_need == 0 or what_you_need == 1:
      #setup query
      query_size = len(query_map.keys())
      query_type = [0]*len(query_ids)
      query_type = torch.tensor([query_type]).to(device)
      query_output, _ = self.bert(query_ids.unsqueeze(0), query_type, query_mask.unsqueeze(0),output_all_encoded_layers=False)
      query_output = query_output.squeeze()
      query_output = self.map_embeddings(query_output, query_map, query_size, device)
      query_idfs_embs = self.get_layer_norm(query_idfs).view(-1,1)
      query_idfs_embs = self.query_idf_NN(query_idfs_embs)
      query_embs = query_idfs_embs + self.get_layer_norm(query_output)
      query_embs = F.relu(self.query_to_boost_NN(query_embs))+1.
      query_to_vocab = self.map_to_new_tensor(q_to_qvocab, query_embs.squeeze(), device)
      query_to_vocab = query_to_vocab.unsqueeze(1)
      if what_you_need == 1:
        return query_to_vocab
    if what_you_need == 0 or what_you_need == 2:
      #setup para
      num_sentences = paragraph_ids.size()[0]
      paragraph_size = [len(p.keys()) for p in para_map]
      paragraph_size_total = np.sum(np.array(paragraph_size))
      para_type = [[1]*paragraph_ids.size()[-1]]*num_sentences
      para_type = torch.tensor([para_type]).to(device).squeeze()
      para_output, _ = self.bert(paragraph_ids, para_type, para_mask,output_all_encoded_layers=False)
      para_output = list(torch.split(para_output, 1, dim=0))
      para_output = [self.map_embeddings(p, para_map[i], paragraph_size[i], device) for i, p in enumerate(para_output)]
      para_scores = [self.para_scores_NN(self.get_layer_norm(p).view(-1,1)) for p in para_scores]
      para_output2 = [para_scores[i] + self.get_layer_norm(p) for i,p in enumerate(para_output)]
      para_output2 = [F.relu(self.para_to_boost_NN(p))+1. for p in para_output2]
      para_output3 = functools.reduce(lambda a,b : torch.cat((a.view(-1),b.view(-1))) ,para_output2)
      if what_you_need == 2:
        return para_output3
    ptokens_to_vocab = torch.sum(torch.stack([self.map_to_new_tensor(p_to_qvocab[i], p.squeeze(), device) for i,p in enumerate(para_output2)]),0)
    idfs_to_vocab = self.map_to_new_tensor(q_to_qvocab, query_idfs.squeeze(), device)
    vocab_scores = ptokens_to_vocab*idfs_to_vocab*(k1+1.)/(ptokens_to_vocab+denom2)
    vocab_scores = vocab_scores.unsqueeze(0)
    logits = torch.mm(vocab_scores, query_to_vocab)/len(query_idfs)-1.
    out = torch.sigmoid(logits)
    print(query_embs.view(-1))
    para_output3 = functools.reduce(lambda a,b : torch.cat((a.view(-1),b.view(-1))) ,para_output2)
    print(para_output3)
    return out.view(1), para_output3, query_embs.view(-1)
    





    #print(para_idfs)
    #para_idfs = para_idfs.cuda()
    #print(para_idfs)
    #para_tfs = torch.stack([self.convert_freq_to_embs(p, self.para_tf_embeddings) for p in para_tfs])
    #self.para_idf_embeddings
    #self.para_tf_embeddings 




    #helper_functions.torch_verbose_print([[nn[0] for nn in n] for n in new_list])







    '''






    #features = is_caps_and_quotes
    #features.required_grad
    #token_probs.requires_grad= False
    num_words = word_embs.size()[0]
    ####para_embs = para_embs.repeat(1,num_words)
    ####para_embs = para_embs.view(1, num_words,-1)
    word_embs = word_embs.unsqueeze(0)
    para_embs = para_embs.unsqueeze(0)
    para_embs = para_embs.unsqueeze(0)
    #word_embs.required_grad
    #para_embs.required_grad
    #word_embs = F.relu(word_embs)
    #para_embs = F.relu(para_embs)
    ####new_embs = word_embs * para_embs + word_embs
    ####new_embs = 
    #new_embs = self.word_embs_layernorm(new_embs)
    ######new_embs = self.multi_head_attention(word_embs, para_embs, para_embs)
    paragraph_attention = self.multi_head_attention(para_embs, word_embs, word_embs)
    self_attention = self.multi_head_attention(word_embs, word_embs, word_embs)
    self_attention = self_attention.sum(-2)
    attention = torch.cat((self_attention, paragraph_attention), 0)
    attention = attention.transpose(-1,-2)
    new_embs = attention
    columns2 = torch.cat((tfs.unsqueeze(0), idfs.unsqueeze(0)), 0)
    columns2 = self.BMtoHeads(columns2.view(-1,2))
    new_embs = columns2 * new_embs/num_words
    #new_embs = self_attention
    #print(new_embs.size())
    #print(new_embs)
    #print(new_embs.size())
    #print(new_embs)
    #new_embs = new_embs.view(12,-1)
    #new_embs = new_embs.sum(0)
    #print(new_embs.size())
    #print(columns2.size())
    #columns2 = columns2 * -1
    #print(columns2.size())
    ########columns2 = self.BMtoHeads(columns.view(-1,1))
    ########new_embs = columns2 * new_embs/num_words
    #print(new_embs.size())
    ########new_embs = torch.cat((new_embs.unsqueeze(0), columns2.unsqueeze(0)),0)
    #print(new_embs.size())
    new_embs = self.NN(new_embs)
    #print(new_embs)
    new_embs = new_embs.transpose(-1,-2)
    #####new_embs = new_embs.view(-1, 9, 768)
    #####new_embs = new_embs.sum(-2)/9
    #new_embs = F.relu(new_embs)
    #new_embs = self.reduce_bert_layers_NN(new_embs)
    #new_embs = new_embs.view(1,-1)
    #new_embs = self.new_embs_layernorm(new_embs)
    ##features = self.features_transform(features)
    #features = self.features_layernorm(features)
    #token_probs = token_probs*100
    #token_probs = token_probs.long()
    #token_probs = token_p_embedding(token_probs)
    #new_embs = (new_embs + token_probs)/ 2
    #token_probs = self.probs_transform(token_probs)
    #token_probs = self.probs_layernorm(token_probs)
    ##features = (token_probs + features)
    ##features = features / 2
    ##new_embs = torch.cat((new_embs, features), 1)
    #new_embs = self.new_embs_layernorm(new_embs)
    #new_embs = new_embs.view(1,-1,self.embedding_size)
    #####columns2 = self.softmax(columns2)
    #####columns2 = columns2.unsqueeze(0).transpose(-1,-2)
    #####new_embs = columns2 + new_embs
    #####new_embs = self.NN(new_embs)
    #####LN = torch.nn.LayerNorm(new_embs.size(), elementwise_affine = False)
    #####new_embs = LN(new_embs)
    #####new_embs = F.sigmoid(new_embs) * 5.
    #new_embs = new_embs.view(1,-1)
    #####new_embs = torch.transpose(new_embs, -1, -2)
    #new_embs = self.softmax(new_embs)
    if is_train:
      columns.required_grad = False
      columns = columns.unsqueeze(0)

      #new_embs = new_embs * 100
      #new_embs = F.relu(new_embs)
      #new_embs = new_embs + 1
      ###weights = torch.mm(new_embs, one_hots)
      #weights = weights.view(1,-1)
      #print(new_embs.size())
      #print(columns.size())
      #print(columns.size())
      out_pre = torch.mm(new_embs, columns.transpose(-1,-2))
      ########LN = torch.nn.LayerNorm(out_pre.size(), elementwise_affine = False)
      ########out_pre = LN(out_pre)
      #print(out.size())
      #print(out)
      out = torch.sigmoid(out_pre)
      #print(out)
      #print(out)
      #print(out.size())
      #print(out.view(-1).size())
      ######out = out.cpu()
      #print(out.size())
      ######out = out.squeeze()
      ######topk = torch.topk(out,2)[1]
      #mask = torch.tensor([i if (i in topk or i in target) else 0 for i in range(out.size()[-1])]).byte()%2
      ######mask = torch.tensor([i if (i in topk or i in target) else 0 for i in range(out.size()[-1])])
      #mask = mask.to(device)
      #out2 = torch.tensor([0. for i in range(out.size()[-1])]).to(device)
      ######out2 = torch.tensor([0. for i in range(out.size()[-1])])
      #print(torch.topk(mask, 5))
      #print(mask.size())
      #print(out.size())
      #print(out2.size())
      #print(torch.topk(out, 5))
      #out2.masked_scatter_(mask, out.contiguous())
      ######out2.scatter_(0, mask, out)
      #print(out2.size())
      #print(torch.topk(out2, 5))
      #out.append(torch.mm(softmax(new_embs), columns))
      ###out = out.view(1,-1)
      return new_embs, out.view(-1), out_pre.view(-1)
    ###else:
    return new_embs
    #return weights
    '''


  def map_embeddings(self, embeddings, query_map, query_size, device):
    embeddings = embeddings.squeeze()
    biggest_mapped_index = query_map[len(query_map)-1]
    #new_list = torch.tensor([[0.0]*self.embedding_size]*biggest_mapped_index)
    new_list = []
    index_counter = [0.0]*biggest_mapped_index
    old_value = 0
    old_index = 0
    list_of_lists_of_tensors=[]
    counter = 0
    for index in query_map.keys():
      value = query_map[index]
      if value!=old_value:
        #print(old_index)
        #print(index)
        #print("")
        #print([t[0].item() for t in torch.narrow(embeddings, 0, old_index, index-old_index ).detach().cpu()])
        list_of_lists_of_tensors.append(torch.narrow(embeddings, 0, old_index, index-old_index ))
        #print(embeddings.size())
        old_index = index
        old_value = value
    #print([t[0].item() for t in torch.narrow(embeddings, 0, old_index, index+1-old_index ).detach().cpu()])
    list_of_lists_of_tensors.append(torch.narrow(embeddings, 0, old_index, index+1-old_index ))
    #print([a.size() for a in list_of_lists_of_tensors])
    list_of_tensors = [a.mean(0) for a in list_of_lists_of_tensors]
    #print([t[0].item() for t in list_of_tensors])
    #print([a.size() for a in list_of_tensors])
    #print(len(list_of_lists_of_tensors))
    #print(len(list_of_tensors))
    return torch.stack(list_of_tensors).to(device)

