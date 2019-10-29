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

#local stuff
from squad_objects2 import *
from helper_functions import *


max_seq_length = 200

#bc = BertClient()
#bc_para = BertClient(port=5557, port_out=5558)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, max_len=max_seq_length)
#note start all sequences with "givemeamap" to get map from new tokens to old tokens
#print(tokenizer.tokenize("givemeamap " + "This is an example of a Bert tokenized sentence."))

output_model_file = "./models/model_file.bin"
output_config_file = "./models/config_file.bin"
output_vocab_file = "./models/vocab_file.bin"


total_correct = 0
total = 0
PATH = 'model_just_para_and_tokens_v28001.pth'
print(PATH)
step = 0
query_text = []
query_weights = []
query_answer = []
is_train = False
data_is_small = False
top_n = 1


list_is_small = False
if data_is_small:
  try:
    list_of_samples = pickle.load(open('small_data_samples.pickle', 'rb'))
    list_is_small = True
  except:
    all_paras = pickle.load(open('../../pickled_squad/All_paras2.pickle', 'rb'))
    print("LOADING ALL PARAS")
else:
  all_paras = pickle.load(open('../../pickled_squad/All_paras2.pickle', 'rb'))
  print("LOADING ALL PARAS")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h_model = load_model2(is_train, device, output_model_file,output_config_file,output_vocab_file)

if __name__ == '__main__':
  if is_train:
    if data_is_small:
      num_epochs = 10
    else:
      num_epochs = 3
    learning_rate = .0001
    if not list_is_small:
      queries = pickle.load(open('../../pickled_squad/training_queries_meta.pickle','rb'))
      queries = queries[0]
      if data_is_small:
        queries = queries[:10]
      list_of_samples = []
    else:
      queries, is_positive, paragraph_sample_index, query_embs, para_embs, token_probs, is_caps_and_quotes, columns,tfs, idfs = list_of_samples[0]
    optimizer = torch.optim.Adam(h_model.parameters(), lr=learning_rate)
    #batch_size = 2
    h_model.cuda()
    average_total = 0.
    average_out = 0.
    for j in range(num_epochs):
      total = 0.
      total_correct = 0.
      sampling_indices = get_random_indices(len(queries))
      #total = 0.
      #total_correct = 0.
      #print(len(queries))
      if data_is_small:
        if j == 1:
          #print(len(list_of_samples))
          pickle.dump(list_of_samples, open('small_data_samples.pickle', 'wb'))
      for i in range(len(queries)):
        step = j * len(queries) + i
        current_index = sampling_indices[i]
        #print(current_index)
        if list_is_small:
          queries, is_positive, paragraph_sample_index, query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs  = list_of_samples[current_index]
        else:
          target = queries[current_index].id
          paragraph_sample_index = target
          paragraph_sample_index = get_random_indices(all_paras.num_docs)[0]
          is_positive = get_random_indices(2)[0]
          paragraph_sample_index = (target if is_positive == 1 else paragraph_sample_index)

        #print("psi" + str(paragraph_sample_index))
        if not data_is_small:
          try:
            query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs = prepare_query2(queries, current_index, token_stats, device, bc, bc_para, all_paras, paragraph_sample_index)
            #solr_weights, out, out_pre = h_model(True, query_embs, para_embs, token_probs, is_caps_and_quotes, columns, device, (average_out-.5), tfs, idfs)
            query_ids, query_masks, sentence_ids, sentence_masks = prepare_query3(query, paragraph, tokenizer, max_sequence_length, device)
            solr_weights, out, out_pre = h_model(query_ids, paragraph_ids, device, tfs, idfs, para_mask=para_mask, query_mask=query_mask, is_train=is_train)
          except:
            print("likely problem with processing query")
            print('')
            continue
        elif not list_is_small:
          query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs= prepare_query2(queries, current_index, token_stats, device, bc, bc_para, all_paras, paragraph_sample_index)
          if j == 0:
            sample = (queries, is_positive, paragraph_sample_index, query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs)
            list_of_samples.append(sample)
          paragraph_sample_index = get_random_indices(len(columns))[0]

          ##solr_weights, out, out_pre = h_model(True, query_embs, para_embs, token_probs, is_caps_and_quotes, columns, device, (average_out-.5), tfs, idfs)
          solr_weights, out, out_pre = h_model(query_ids, paragraph_ids, device, tfs, idfs, para_mask=para_mask, query_mask=query_mask, is_train=is_train)
        #loss = cal_loss(out, target, True)
        #print(target.size())
        #print(out)
        target = 1. if is_positive == 1 else 0.
        target = torch.tensor([target], device=device)
        loss = F.soft_margin_loss(out, target)
        #loss = F.cross_entropy(out, target, reduction='sum')
        loss.backward()
        #print(loss)
        optimizer.step()
        optimizer.zero_grad()
        if i%1000==1 and not data_is_small:
          #torch.save(h_model.state_dict(), str("model_just_para_and_tokens_v" + str(step) + ".pth"))
          torch.save(model.state_dict(), output_model_file)
          model.config.to_json_file(output_config_file)
          tokenizer.save_vocabulary(output_vocab_file)
        if (step % 100 == 0 and len(queries[current_index].tokens) > 1) or data_is_small:
          total_correct, total = print_statistics2(step, loss, total_correct, total, is_train, solr_weights.cpu().detach(), queries[current_index], out, target)
          if total_correct/total > .5:
            pass
          else:
            average_total += 1
            average_out = ((average_out * (average_total-1) + out_pre.detach().item()) / average_total)
          #print(average_out)
  else:
    h_model.cuda()
    queries = pickle.load(open('../../pickled_squad/testing_queries_meta.pickle','rb'))
    queries = queries[0]
    token = torch.tensor([1], device = device)
    for i in range(len(queries)):
      query_embs, para_embs, token_probs, is_caps_and_quotes, columns, tfs, idfs = prepare_query2(queries, i, token_stats, device, bc, bc_para, all_paras, 0)
      target = torch.tensor([queries[i].id], device=device)
      if True:
        #try:
         #solr_weights = h_model(False, query_embs, para_embs, columns, device,tfs, idfs)
         solr_weights, out, out_pre = h_model(query_ids, paragraph_ids, device, tfs, idfs, para_mask=para_mask, query_mask=query_mask, is_train=is_train)

      '''except:
        print("likely problem with processing query")
        print('')
        continue'''
      #total_correct, total = print_statistics(None, None, total_correct, total, is_train, solr_weights, queries[i], top_n, out)
      if i < 200:
        print(solr_weights)
      if i % 2000 == 0 or i%len(queries) == 0:
        print(i)
        query_weights, query_text, query_answer = save_stats_and_dump(solr_weights, queries[i], query_weights, query_text, query_answer, dump = True)
      else:
        query_weights, query_text, query_answer = save_stats_and_dump(solr_weights, queries[i], query_weights, query_text, query_answer, dump = False)

