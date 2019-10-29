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
import numpy as np
import sys
from string import punctuation
import unicodedata


#local stuff
from squad_objects2 import *
from TokenStats import *

#from helper_functions import *
from hybrid_model import BertBoosting

#Bert stuff
sys.path.append('../../../../BERT-pytorch/pytorch-pretrained-BERT/')
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)




#paragraphs = pickle.load(open('../../pickled_squad/para_encodings.pickle','rb'))
#paragraphs2 = pickle.load(open('../../pickled_squad/val_para_encodings.pickle', 'rb'))
#paragraphs += paragraphs2

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def f(para): 
  tokens = para.para.split()
  size = len(tokens)
  dictionary = {}
  length = 0
  for token in tokens:
    if token in dictionary.keys():
      dictionary[token] += 1 
    else:
      dictionary[token] = 1
    length+=1
  
  return Metatext(size, dictionary, para.para_id)

def combine_dictionaries_for_idf(dictionary1, dictionary2):
  temp_dictionary = {}
  for key in dictionary1.keys():
    if key in dictionary2.keys():
      temp_dictionary[key] = dictionary1[key] + 1
    else:
      temp_dictionary[key] = dictionary1[key]
  for key in dictionary2.keys():
    if key not in dictionary1.keys():
      temp_dictionary[key] = 1
  return temp_dictionary 

def g(list_of_Metas):
    total_dictionary = {}
    total_size = 0
    for p in list_of_Metas:
      total_dictionary = combine_dictionaries_for_idf(total_dictionary, p.dic)
      total_size += p.size
    return Metatext(total_size, total_dictionary, None)

def make_column(para, word_list, idfs, avg_length):
  doc_column = []
  for word in word_list:
    if word in para.dic.keys():
      frequency = para.dic[word]
      idf = idfs[word]
      nom = frequency * (k1 + 1.)
      denom = frequency + k1 * (1. - b + b * para.size/avg_length)
      cell_total = idf * nom / denom
      doc_column.append(cell_total)
    else:
      doc_column.append(0.)
  return (para.id, np.array(doc_column))

def get_idfs(word_list, all_paras):
  return [all_paras.idfs[word] if word in all_paras.global_dictionary.keys() else 0.0 for word in word_list]
def get_avg_tf(word_list, all_paras):
  return [np.average(all_paras.get_column(word)) if word in all_paras.global_dictionary.keys() else 0.0 for word in word_list]
def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        pred = pred.view(1,-1)
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        #print(one_hot)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')

    return loss

def get_random_indices(num_training_samples):
  all_indices = [i for i in range(num_training_samples)]
  sampling_indices = random.sample(all_indices, k=num_training_samples)
  return sampling_indices

def check_accuracy(out, paragraph_num, is_talky, top_n):
  top_five = np.argsort(-out.squeeze().cpu().detach().numpy())[:top_n]
  if paragraph_num in top_five:
    if is_talky:
      print("Got one right!")
    return 1
  else:
    if is_talky:
      print(str(paragraph_num) + "is not in " + str(top_five))
    return 0
'''
def construct_one_hots(test_query, all_paras):
  test_dictionary = {}
  one_hots = []
  for t in test_query.tokens:
    t = re.sub('\?$','',t)
    one_hot = []
    for token in all_paras.global_dictionary.keys():
      if t==token:
        one_hot.append(1.)
      else:
        one_hot.append(0.)
    one_hots.append(one_hot)
  return torch.tensor(one_hots)
'''
def torch_verbose_print(t):
  torch.set_printoptions(profile="full")
  print(t)
  torch.set_printoptions(profile="default")

def get_caps_and_quotes(question):
  is_in_quotes = False
  caps_vector = []
  quotes_vector = []
  for index, s in enumerate(question.split()):
    try:
      if s[0].isupper() and index !=0 and len(s) > 3:
        #print("Is caps")
        caps_vector.append(2.)
      else:
        caps_vector.append(1.)
      if s[-1]=="\"":
        is_in_quotes = False
        #print("Is in quotes")
        quotes_vector.append(2.)
      elif s[0]=="\"" or is_in_quotes:
        is_in_quotes = True
        #print("is in quotes")
        quotes_vector.append(2.)
      else:
        quotes_vector.append(1.)
    except:
      print("Unable to process features")
      return [[1.,1.]]*len(question.split())
  return list(zip(caps_vector, quotes_vector))


'''def prepare_query(all_paras, queries, index, token_stats, is_dev, gpu, bc, bc_para):
  device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
  query = queries[index]
  if not is_dev:
    one_hots = construct_one_hots(query, all_paras)
  else:
    one_hots = [1.,0.,0.]

  try:
    query_embs = query.get_embs_tokens(bc)
    para_embs = query.get_para_embs(bc_para)
    token_probs = [[token_stats.get_token_precision(q),token_stats.get_token_recall(q),token_stats.get_token_f1(q)] for q in query.tokens]
    is_caps_and_quotes = get_caps_and_quotes(query.question)
  except:
    print("UNABLE to process query: " + str(query.question))
    return (None, None, None, None, None, None)
  one_hots = torch.tensor(one_hots, device=device)
  query_embs = torch.tensor(query_embs, device = device)
  para_embs = torch.tensor(para_embs, device=device)
  para_embs = para_embs.squeeze() 
  token_probs = torch.tensor(token_probs, device=device)
  is_caps_and_quotes = torch.tensor(is_caps_and_quotes, device=device)
  query_id = torch.tensor([query.id], device=device)
  return one_hots, query_embs, para_embs, token_probs, is_caps_and_quotes, query_id
'''
def find_in_list(t, a_list, max_value):
  try:
    return a_list.index(t)
  except:
    #print("unable to find " + str(t) + " in query")
    return max_value

def prepare_query2(queries, index, token_stats, device, bc, bc_para, all_paras, paragraph_sample_index):
  query = queries[index]
  try:
    query_embs = query.get_embs_tokens(bc)
    para_embs = query.get_para_embs(bc_para)
    token_probs = [[token_stats.get_token_precision(q),token_stats.get_token_recall(q),token_stats.get_token_f1(q)] for q in query.tokens]
    is_caps_and_quotes = get_caps_and_quotes(query.question)
    columns = all_paras.get_columns(query.tokens)
    columns = [c[paragraph_sample_index] for c in columns]
  except:
    print("UNABLE to process query: " + str(query.question))
    #return (None, None, None, None, None, None)
  tfs = get_avg_tf(query.tokens, all_paras)
  #print(tfs)
  idfs = get_idfs(query.tokens, all_paras)
  #print(idfs)

  query_embs = torch.tensor(query_embs, device = device)
  para_embs = torch.tensor(para_embs, device=device)
  para_embs = para_embs.squeeze() 
  token_probs = torch.tensor(token_probs, device=device)
  is_caps_and_quotes = torch.tensor(is_caps_and_quotes, device=device)
  columns = torch.tensor(columns, device = device).float()
  tfs = torch.tensor(tfs, device = device).float()
  idfs = torch.tensor(idfs, device = device).float()
  return query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs

def prepare_query3(tokenizer, max_seq_length, device, what_you_need, paragraph=None, query=None):
  if what_you_need == 0 or what_you_need==1:
    query = "givemeamap " + query
    query_ids, query_map = tokenizer.tokenize(query)
    query_ids = tokenizer.convert_tokens_to_ids(query_ids)
    query_masks = [1]*len(query_ids)
    while len(query_masks)<max_seq_length:
      query_masks.append(0)
      query_ids.append(0)
    assert(len(query_ids)==max_seq_length)
    assert(len(query_masks)==max_seq_length)
    if what_you_need == 1:
      return [torch.tensor(a).to(device) for a in [query_ids, query_masks]], query_map
  if what_you_need == 0 or what_you_need==2:
    sentences = ["givemeamap "+s for s in paragraph if len(s) > 0]
    sentence_tokens, sentence_maps = zip(*[tokenizer.tokenize(s) for s in sentences])
    sentence_ids = [tokenizer.convert_tokens_to_ids(s) for s in sentence_tokens]
    sentence_masks = [[1]*len(s) for s in sentence_ids]
    for i,s in enumerate(sentence_ids):
      while len(sentence_masks[i]) < max_seq_length:
        sentence_masks[i].append(0)
        sentence_ids[i].append(0)
    for i,s in enumerate(sentence_ids):
      assert(len(sentence_masks[i]) == max_seq_length)
      assert(len(sentence_ids[i]) == max_seq_length)
    if what_you_need == 2:
      return [torch.tensor(a).to(device) for a in [sentence_ids, sentence_masks]], (sentence_maps)
    else:
      return [torch.tensor(a).to(device) for a in [query_ids, query_masks, sentence_ids, sentence_masks]], (query_map, sentence_maps)
def dictionary_lookup(dictionary, token):
  try:
    return 1.0/dictionary[token]
  except:
    print("Could not find " + str(token) + " in dictionary")
    return 0.0

def lookup_idfs(token_list, idfs):
  return [ [dictionary_lookup(idfs, token) for token in sequence ] for sequence in token_list]

def lookup_tfs(token_list, tfs, para_num):
  outer = []
  for sequence in token_list:
    inner = []
    for token in sequence:
      try:
        inner.append(tfs[para_num][token])
      except:
        inner.append(0.0)
        #print("unable to find " + str(token) + " in paragraph " + str(para_num))
    outer.append(inner)
  return outer
def compute_scores(idfs, tfs, paragraph_num, paragraph, para_lengths, k1, b, para_average):
  para_size = para_lengths[paragraph_num]
  para_idfs = np.array(lookup_idfs([p.split() for p in paragraph], idfs))
  para_tfs = np.array([np.array(l,dtype=np.float) for l in lookup_tfs([p.split() for p in paragraph], tfs, paragraph_num)])
  nom = np.array(para_tfs) * (k1+1.)
  denom2 = k1 * (1. - b + b * para_size/para_average)
  denom = np.array(para_tfs) + denom2
  para_scores = para_idfs * nom / denom
  return para_scores, denom2

def compute_tensor_scores(idfs, tfs, paragraph, para_lengths, k1, b, parag_average):
  para_size = para_lengths[paragraph_num]
  para_idfs = np.array(lookup_idfs([p.split() for p in paragraph], idfs))
  para_tfs = np.array([np.array(l,dtype=np.float) for l in lookup_tfs([p.split() for p in paragraph], tfs, paragraph_num)])
  nom = np.array(para_tfs) * (k1+1.)
  denom = np.array(para_tfs) + k1 * (1. - b + b * para_size/para_average)
  para_scores = para_idfs * nom / denom
  return para_idfs, para_tfs, para_scores


def prepare_sentence(sentence, bc, bc_para, device, all_paras):
  sentence, tokens = process_sentence(sentence)
  token_embs = get_embs_tokens(bc, sentence, tokens)
  sentence_embs = get_sentence_embs(bc_para, sentence)
  tfs = get_avg_tf(tokens, all_paras)
  idfs = get_idfs(tokens, all_paras)

  token_embs = torch.tensor(token_embs, device = device)
  sentence_embs = torch.tensor(sentence_embs, device=device)
  sentence_embs = sentence_embs.squeeze()

  tfs = torch.tensor(tfs, device = device).float()
  idfs = torch.tensor(idfs, device = device).float()

  return sentence, tokens, token_embs, sentence_embs, tfs, idfs

def process_sentence(sentence):
  original_tokens = [re.sub('\?$', '', token) for token in sentence.split()]
  original_tokens = [''.join([e for e in token if e.isalnum()]) for token in original_tokens]
  sentence = ' '.join(original_tokens)
  return sentence, original_tokens

def get_BERT_tokens(bc, sentence):
  sentence, original_tokens = process_sentence(sentence)
  bertized_query = bc.encode([sentence], show_tokens=True)
  new_tokens = bertized_query[1][0]
  return new_tokens

def filter_BERT_sentence(BERT_tokens):
  #return [re.sub('^##|\?$','',e) for e in BERT_tokens if e not in ['[CLS]', '[SEP]', '0_PAD', '[UNK]']]
  return [ e for e in BERT_tokens if e not in ['[CLS]', '[SEP]', '0_PAD', '[UNK]']]

def get_embs_tokens(bc, sentence, original_tokens):
  #print(original_tokens)
  #print(sentence)
  bertized_query = bc.encode([sentence], show_tokens=True)
  tokens = [re.sub('^##|\?$','',e) for e in bertized_query[1][0] if e not in ['[CLS]', '[SEP]', '0_PAD']]
  #print(tokens)
  embeddings = bertized_query[0]
  token_pointer = 0
  original_token_embeddings = []
  embeddings = np.squeeze(embeddings)
  for index, o_token in enumerate(original_tokens):
    #print("o_token" + str(o_token))
    counter = 1
    embedding = np.array(embeddings[token_pointer])
    token_combo = tokens[token_pointer]
    while o_token != token_combo and (token_pointer+1)< len(tokens) and token_combo!='[UNK]':
      token_pointer += 1
      embedding += np.array(embeddings[token_pointer])
      token_combo += tokens[token_pointer]
      #print(token_pointer)
      #print(len(tokens))
      #print(token_combo)
      counter += 1
    token_pointer += 1
    #print(token_pointer)
    #print("TEST")
    embedding/=counter
    #print("TEST")
    original_token_embeddings.append(embedding)
    #print("TEST")

  return original_token_embeddings

def get_sentence_embs(bc, sentence):
  return bc.encode([sentence])

def clean_text(text):
  return unicodedata.normalize('NFKD', text.translate(str.maketrans('', '', punctuation)).lower()).encode('ascii','ignore').decode('utf-8')

def print_statistics(step, loss, total_correct, total, is_train, solr_weights, query, top_n, out):
  if is_train:
    print("Iteration number: " + str(step))
    print("loss: " + str(loss))
  print("tokens: " + str(query.tokens))
  solr_weights = solr_weights.squeeze()
  #solr_weights.cpu()
  print(list(zip(query.tokens, solr_weights.tolist())))
  total_correct = total_correct + check_accuracy(out, query.id, True, top_n)
  total = total + 1
  print("Accuracy so far " + str(float(total_correct)/total))
  return total_correct, total

def print_statistics2(step, loss, total_correct, total, is_train, solr_weights, query, out, target):
  if is_train:
    print("Iteration number: " + str(step))
    print("loss: " + str(loss))
  print("tokens: " + str(query.tokens))
  solr_weights = solr_weights.squeeze()
  #solr_weights.cpu()
  print(list(zip(query.tokens, solr_weights.tolist())))
  if (out.item() <= 0 and target.item() <= 0) or (out.item() > 0 and target.item() > 0):
    total_correct += 1.
    print(" ")
    print("GOT ONE RIGHT!")
    print(" ")
  total += 1.
  print("Accuracy so far " + str(float(total_correct)/total))
  return total_correct, total


class gpu_chooser(object):
  def __init__(self, device_num):
    choice = "cuda:" + str(device_num)
    self.device = torch.device(choice if torch.cuda.is_available() else "cpu")
  def receive(self, list_to_send):
    return [a.to(self.device) for a in list_to_send]


def load_model(is_train, device, list_is_small, PATH):
  #gpu = gpu_chooser(3)
  model = hybridModel(768, config)
  '''try:
    model=BertBoosting(config)
  except:
    model = BertBoosting.from_pretrained("bert-base-cased")
  '''
  if os.path.exists(PATH):
    print("loading model...")
    model.load_state_dict(torch.load(PATH))
  if is_train:
    model.train()
  else:
    model.eval()
  model = model.cuda()
  if not list_is_small:
    token_stats = pickle.load(open('../../pickled_squad/token_stats.pickle','rb'))
    return model, token_stats
  return model, None

def load_model2(is_train, device, output_model_file,output_config_file,output_vocab_file, max_seq_length,chooser):
  output_config_file= output_config_file+str(chooser)+".bin"
  output_model_file = output_model_file+str(chooser)+".bin"
  output_vocab_file = output_vocab_file+str(chooser)+".bin"
  try:
    config = BertConfig.from_json_file(output_config_file)
    model = BertBoosting(config,768)
    state_dict = torch.load(output_model_file)
    model.load_state_dict(state_dict)
    tokenizer = BertTokenizer(output_vocab_file, do_lower_case=False,max_len=max_seq_length)
    model.cuda()
  except:
    print("could not load file, initializing randomly")
    model = BertBoosting.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, max_len=max_seq_length)
  if is_train == 0:
    model.train()
  else:
    model.eval()
  return model, tokenizer
def save_stats_and_dump(solr_weights, query, query_weights, query_text, query_answer, dump):
  query_weights.append(solr_weights.detach().tolist())
  #print(solr_weights)
  #print(query.tokens)
  query_text.append(query.tokens)
  query_answer.append(query.id)
  if dump:
    pickle_data = (query_text, query_weights, query_answer)
    pickle.dump(pickle_data, open('../../pickled_squad/test_weights.pickle', 'wb'))
  return query_weights, query_text, query_answer

def construct_one_hots(test_query, all_paras):
  test_dictionary = {}
  one_hots = []
  for t in test_query.tokens:
    #print(t)
    t = re.sub('\?$','',t)
    one_hot = []
    for token in all_paras.global_dictionary.keys():
      if t==token:
        one_hot.append(1.)
      else:
        one_hot.append(0.)
    one_hots.append(one_hot) 
  return one_hots


#NOTE: BELOW IS JUST UNUSED COMMENTS FROM MODEL


'''
    print(queries[1].dic)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    test_matrix = torch.Tensor(all_paras.matrix)
    print(all_paras.calculate_score(queries[1]))
    test_matrix.to(device)
    counts = []
    for word in all_paras.global_dictionary.keys():
     try:
       counts.append(queries[1].dic[word])
     except:
       counts.append(0)
    counts = torch.Tensor(np.array([counts]))
    counts.to(device)
    print(torch.mm(counts, test_matrix)[0])
    '''
    



    
'''

  except:
    paragraphs = pickle.load(open('../../pickled_squad/para_encodings.pickle','rb'))
    paragraphs2 = pickle.load(open('../../pickled_squad/val_para_encodings.pickle', 'rb'))
    paragraphs += paragraphs2
    print("UNABLE TO FIND MATRIX FILE; REPRODUCING")
    k1 = 1.5
    b = .75
    paragraphs = p.map(f, paragraphs)
    print(paragraphs[0].dic)
    chunk_size = int(len(paragraphs)/40)
    list_of_lists = list(chunks(paragraphs, chunk_size))
    print(list_of_lists[0][0].dic)
    chunked_stats = p.map(g, list_of_lists)
    final_stats = g(chunked_stats)
    all_paras = All_para_meta(final_stats, paragraphs, k1, b, p)
    pickle.dump(all_paras, open('All_para_meta.pickle', 'wb'), protocol=4)

'''
#paragraphs = squad_paras(paragraphs, k1, b, bc)


'''
print(len(paragraphs))

paragraphs = [p for p in paragraphs[:1000]]
print(paragraphs[1].para_tokens)
#paragraphs = [p for p in paragraphs[:100000]] #.para_id for id, .para for para, .para_tokens for tokens

#p0 = squad_para(0, 'this is a a sample'.split(' '))
#p1 = squad_para(1, 'this is another another example example example'.split(' '))
#p2 = squad_para(2, 'final doc here here'.split(' '))

#query = squad_question(1,False, 'a query example', 1, 'query', 1, 1)

#the_list = [p0, p1, p2]
k1 = 1.5
b = .75
#paragraphs = squad_paras(the_list, k1, b)

paragraphs = squad_paras(paragraphs, k1, b, bc)
print(paragraphs.list_of_paras[1].para_tokens)

queries = pickle.load(open('../../pickled_squad/questions_conv.pickle','rb'))
queries = [q for q in queries if q.paragraph_num in paragraphs.get_ids()] #attributes: .paragraph_num, .question, .is_impossible
[q.re_init(bc) for q in queries]
predictions = [paragraphs.calculate_score(q) for q in queries]
reality = [q.paragraph_num for q in queries]
correct = 0.
total = 0.
for i, p in enumerate(predictions):
  if reality[i] == p:
    correct += 1
  total += 1
print(correct/total)

'''
'''
paragraphs.re_init(k1,b)
print("finished processing")

queries = pickle.load(open('../../../pickled_squad/questions_conv.pickle','rb'))
queries = [q for q in queries if q.paragraph_num in paragraphs.get_ids()] #attributes: .paragraph_num, .question, .is_impossible
[q.re_init() for q in queries]
print([paragraphs.calculate_score(q) for q in queries])
print([q.paragraph_num for q in queries])

'''
