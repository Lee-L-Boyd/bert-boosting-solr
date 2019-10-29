
import pickle
import os
import re
from multiprocessing import Pool
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from bert_serving.client import BertClient

#Bert stuff
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

#local stuff
from squad_objects2 import *
from helper_functions import *

#solr stuff
import urllib
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2


max_seq_length = 512

output_model_file = "./models/model_file"
output_config_file = "./models/config_file"
output_vocab_file = "./models/vocab_file"
chooser = 277

total_correct = 0
total = 0
top_n = 1

query_text = []
query_weights = []
query_answer = []

step = 0
what_you_need = 1
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h_model,tokenizer = load_model2(what_you_need, device, output_model_file,output_config_file,output_vocab_file,max_seq_length,chooser)
h_model.cuda()
queries = pickle.load(open('../../pickled_squad/training_queries_meta.pickle','rb'))[0]

paragraphs = pickle.load(open('../../pickled_squad/clean_para_id_plus_tok.pickle', 'rb'))
tfs = pickle.load(open('../../pickled_squad/clean_tfs.pickle', 'rb'))
idfs = pickle.load(open('../../pickled_squad/clean_idfs.pickle', 'rb'))

para_lengths = [np.sum(np.array([len(p.split()) for p in para])) for para in paragraphs]
para_average = np.average(para_lengths) 

k1=1.2
b=.75
def get_para_stuff(i, paragraphs, para_lengths):
  return paragraphs[i], para_lengths[i], [p.split() for p in paragraphs[i]]

def get_query_stuff(query):
  q_tokens = query.split()
  query_idfs = torch.tensor(lookup_idfs([q_tokens], idfs)).to(device).double()
  qvocab = list(set(q_tokens))
  q_to_qvocab = [qvocab.index(q) for q in q_tokens]
  max_value = max(q_to_qvocab)
  return q_to_qvocab, query_idfs, qvocab, max_value

def print_info(target,out,correct,total,num_queries, i):
  print("target " + str(target))
  print("predicted " + str(out))
  if (target == 1. and target - out.squeeze().item() < .5) or (target == 0. and target + out.squeeze().item() < .5):
    correct += 1
    print("GOT ONE RIGHT!")
  total += 1
  if num_queries-1==i:
    print("ACCURACY " + str(correct/total))
  return correct, total 

total_this_epoch = 0.
correct_this_epoch = 0.

queries = queries[:10]
num_queries = len(queries)
sampling_indices_query = get_random_indices(len(queries))
sampling_indices_paragraph = get_random_indices(len(paragraphs))

num_epochs = 7

optimizer = torch.optim.Adam(h_model.parameters(), lr=.001)
if what_you_need == 0:
  for e in range(num_epochs):
    for i in range(num_queries):
      step = e*num_queries + i + chooser
      current_index = sampling_indices_query[i]
      query = clean_text(queries[current_index].question)
      para_id=queries[current_index].id
      target = get_random_indices(2)[0]*1.
      para_num = para_id if (target == 1. or e*i < 10) else sampling_indices_paragraph[i]
      paragraph, para_size, p_tokens = get_para_stuff(para_num,paragraphs, para_lengths)
      q_to_qvocab, query_idfs, qvocab, max_value = get_query_stuff(query)
      p_to_qvocab = [[find_in_list(t, qvocab, max_value) for t in s] for s in p_tokens]
      [query_ids, query_masks, sentence_ids, sentence_masks], (query_map, sentence_maps) = prepare_query3(tokenizer, max_seq_length, device, what_you_need,paragraph=paragraph,query=query)
      #NOTE: denom2 is just the part of BM25 that doesn't change once you know k1, b, paralengths, and avgparalengths (it can be moved out of the loop for more efficiency)
      para_scores,denom2 = compute_scores(idfs, tfs, para_num, paragraph, para_lengths, k1, b, para_average)
      para_scores = [torch.tensor(p).double().to(device) for p in para_scores]
      out, para_output, query_embs = h_model(device, q_to_qvocab=q_to_qvocab, p_to_qvocab=p_to_qvocab, denom2=denom2, para_scores=para_scores, query_idfs=query_idfs, query_ids=query_ids, \
                                  paragraph_ids=sentence_ids, para_mask=sentence_masks, query_mask=query_masks, what_you_need=what_you_need, query_map = query_map, para_map = sentence_maps )
      correct_this_epoch, total_this_epoch = print_info(target, out, correct_this_epoch, total_this_epoch,num_queries,i)
      target = torch.tensor([target], device=device).float()
      loss = F.soft_margin_loss(out, target)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if i%num_epochs==1:
        #torch.save(h_model.state_dict(), str("model_just_para_and_tokens_v" + str(step) + ".pth"))
        torch.save(h_model.state_dict(), output_model_file+str(step)+".bin")
        h_model.config.to_json_file(output_config_file+str(step)+".bin")
        tokenizer.save_vocabulary(output_vocab_file+str(step)+".bin")

else:
  h_model.eval()
  query_text = []
  query_weights = []
  query_answer = []
  test_queries = pickle.load(open('../../pickled_squad/testing_queries_meta.pickle','rb'))[0]
  for i in range(len(test_queries)):
    current_index = i
    query1 = test_queries[current_index]
    query = clean_text(test_queries[current_index].question)
    print(query)
    query1.tokens = query.split()
    q_to_qvocab, query_idfs, q_vocab, max_value = get_query_stuff(query)
    [query_ids, query_masks], (query_map) = prepare_query3(tokenizer, max_seq_length, device, 1, query=query)
    query_embs = h_model(device, q_to_qvocab=q_to_qvocab, query_idfs=query_idfs, query_ids=query_ids, query_mask=query_masks, what_you_need=1, query_map = query_map)
    if i % 2000 == 0 or i%len(queries) == 0:
        print(i)
        query_weights, query_text, query_answer = save_stats_and_dump(query_embs.view(1,-1), query1, query_weights, query_text, query_answer, dump = True)
    else:
      query_weights, query_text, query_answer = save_stats_and_dump(query_embs.view(1,-1), query1, query_weights, query_text, query_answer, dump = False)


