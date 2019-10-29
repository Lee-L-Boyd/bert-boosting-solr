import pickle
from xml.sax.saxutils import escape
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
import functools
from bert_serving.client import BertClient

#local stuff

from squad_objects2 import *
from helper_functions import *
PATH = 'model_just_para_and_tokens_v28001.pth'
is_train = False


bc = BertClient()
#bc_para = BertClient(port=5557, port_out=5558)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#all_paras = pickle.load(open('../../pickled_squad/All_paras2.pickle', 'rb'))
tokenized_paras = pickle.load(open('../../pickled_squad/All_paras_tokenized_list.pickle', 'rb'))


#h_model, token_stats = load_model(is_train, device, True, PATH)

batch_size = 1000
f = open('../../xmled_squad/tokenized_paras.xml', 'w')
f.write('<add>\n')

def make_para_xml(f, para_id, text):
  f.write('<doc>\n')
  f.write('<field name="para_id" type="uuid" indexed="true">' + escape(str(para_id)) + '</field>\n')
  f.write('<field name="text" type="text">' + escape(str(text)) + '</field>\n')
  f.write('</doc>\n')


def make_paras_xml(f, paragraphs):
  string_to_write = ' '
  for i in range(len(paragraphs)):
    string_to_write += '<doc>\n' + \
      '<field name="para_id" type="uuid" indexed="true">' + escape(str(paragraphs[i][0])) + '</field>\n' + \
      '<field name="text" type="text">' + escape(str(paragraphs[i][1])) + '</field>\n' + \
      '</doc>\n'
  f.write(string_to_write)

new_paragraph_text = ''
paragraphs = []
for index, para in enumerate(tokenized_paras):
  for sentence_index, st in enumerate(para.sentence_tokens):
    try:
      #if True:
      #print(st)
      tokens = get_BERT_tokens(bc, st)
      tokens = filter_BERT_sentence(tokens)
      new_paragraph_text += ' ' + ' '.join(tokens)
    except:
      print("Unable to process sentence: " + str(st))
  paragraph_w_id = (para.id, new_paragraph_text)
  paragraphs.append(paragraph_w_id)
  new_paragraph_text = ''
  if (index % batch_size == 0 and index != 0) or index == (len(tokenized_paras)-1):
    print(paragraphs)
    make_paras_xml(f, paragraphs)
    paragraphs = []
    print("batch written " + str(index))
f.write('</add>')
f.close()



'''


     try:
      sentence, tokens, token_embs, sentence_embs, tfs, idfs = prepare_sentence(st, bc, bc_para, device, all_paras)
      solr_weights = h_model(is_train, token_embs, sentence_embs, None, device,tfs,idfs)
      solr_weights = solr_weights.squeeze()
      solr_weights = solr_weights * 10
      solr_weights = [int(s) for s in solr_weights.tolist()]
      boosted_words = zip(solr_weights, tokens)
      boosted_words = [ [ token[1] ]*token[0] for token in boosted_words ]
      boosted_words = functools.reduce(lambda a,b : a+b,list(boosted_words))
      boosted_words = ' '.join(boosted_words)
      new_paragraph_text += boosted_words
    except:
      print( "Unable to boost sentence " +str(sentence_index) + " in paragraph " + str(index) )
      new_paragraph_text += sentence
  paragraph_w_id = (para.id, new_paragraph_text)
  paragraphs.append(paragraphs_w_id)
  new_paragraph_text = ''
  if (index % batch_size == 0 and index != 0) or index == (len(tokenized_paras)-1):
    make_paras_xml(f, paragraphs)
    paragraphs = []
    print("batch written " + str(index))'''







#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#list_is_small = True

#h_model, token_stats = load_model(is_train, device, list_is_small, PATH)


#[p.tokenize() for p in paras_old[:1]]
#id = all_paras.list_of_paras[0].id

#print(id)
#print([p.para_tokens for p in paras_old if p.para_id == id])
#print([(p.id, len(p.dic.keys())) for p in all_paras.list_of_paras[:1]])



#pickle.dump(, open('tiny_paras.pickle', 'wb'))


#query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs = prepare_query2(queries, i, token_stats, device, bc, bc_para, all_paras, 0)
