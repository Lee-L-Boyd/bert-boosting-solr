from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
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

#~/BERT-pytorch/pytorch-pretrained-BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_length = 200

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, max_len=max_seq_length)
print(tokenizer.tokenize("givemeamap " + "This is an example of a Bert tokenized sentence."))
print(tokenizer.tokenize("This is an example of a Bert tokenized sentence."))

ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("This is an example of a Bert tokenized sentence."))
input_mask = [1]*len(ids)
while len(ids) < max_seq_length:
  ids.append(0)
  input_mask.append(0)
#NOTE:can put other information instead of segment ids here
segment_ids = [0]*len(ids)

assert len(ids) == max_seq_length
assert len(input_mask) == max_seq_length
assert len(segment_ids) == max_seq_length

ids = torch.tensor([ids]).to(device)
input_mask = torch.tensor([input_mask]).to(device)
segment_ids = torch.tensor([segment_ids]).to(device)

print(ids)



#class BertForTokenClassification(BertPreTrainedModel):
''''''
class BertBoosting(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertBoosting, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.NN=nn.Linear(6,1) 
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        ''''''
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        print(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


model = BertBoosting.from_pretrained("bert-base-cased")
''''''
model.cuda()
model.eval()
model(ids, token_type_ids = segment_ids, attention_mask=input_mask)
''''''
ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("This is an example of a Bert tokenized sentence."))
input_mask = [1]*len(ids)
while len(ids) < max_seq_length:
  ids.append(0)
  input_mask.append(0)
#NOTE:can put other information instead of segment ids here
segment_ids = [0]*len(ids)

assert len(ids) == max_seq_length
assert len(input_mask) == max_seq_length
assert len(segment_ids) == max_seq_length

ids = torch.tensor([ids]).to(device)
input_mask = torch.tensor([input_mask]).to(device)
segment_ids = torch.tensor([segment_ids]).to(device)



##
model(ids, token_type_ids = segment_ids, attention_mask=input_mask)
output_model_file = "./models/model_file.bin"
output_config_file = "./models/config_file.bin"
output_vocab_file = "./models/vocab_file.bin"
torch.save(model.state_dict(), output_model_file)
model.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_vocab_file)


##
config = BertConfig.from_json_file(output_config_file)
model = BertBoosting(config)
state_dict = torch.load(output_model_file)
model.load_state_dict(state_dict)
tokenizer = BertTokenizer(output_vocab_file, do_lower_case=False,max_len=max_seq_length)
model.cuda()
model.eval()
model(ids, token_type_ids = segment_ids, attention_mask=input_mask)

'''
top_model_file = ".top_of_bert.pth"
bert_file = ".bert_model.pth"
try:
  model.load_state_dict(torch.load(top_model_file))
except:
  print("unsuccessful model load")
model.cuda()
model(ids, token_type_ids = segment_ids, attention_mask=input_mask)


model.saveBert(bert_file)
torch.save(model.state_dict(), top_model_file)
'''
