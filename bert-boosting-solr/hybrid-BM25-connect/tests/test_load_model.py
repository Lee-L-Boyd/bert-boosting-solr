import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torch_multi_head_attention import MultiHeadAttention
import helper_functions
from hybrid_model import hybridModel
from paragraph_booster import ParaBoost

model = ParaBoost(768)
model_dict = model.state_dict()
#model2 = hybridModel(768)
original_model_dict = torch.load('model_just_para_and_tokens_v17001.pth')
pretrained_dict = {k:v for k,v in original_model_dict.items() if k in model_dict}
print(pretrained_dict)
model.load_state_dict(pretrained_dict)

def get_embeddings(paras, embedding_size):
  
  sentence_embeddings = [] #the list of all the sentence embeddings 
  paragraph_embeddings = [] #the list of all paragraph embeddings

  paragraph_lookup = {} #stores all the sentences indices (values) from sentence_embeddings forr the paragraph (key)
  sentence_lookup = {} #stores the paragraph index (value) from paragraph_embeddings forr the sentence (key)

  num_paras = len(paras)
  sentence_counter = 0
  previous_i = 0
  paragraph_total = 1
  for i in range(num_paras):
    paragraph_embeddings.append(np.zeros(embedding_size))
    paragraph_lookup[i] = []
    tp = paras[i].embedding_list
    for sentence in list(tp[0]):
      if previous_i != i:
        paragraph_embeddings[previous_i]/=paragraph_total
        paragraph_total = 1
        previous_i = i
      sentence_embeddings.append(sentence)
      sentence_lookup[sentence_counter] = i
      paragraph_lookup[i].append(sentence_counter)
      sentence_counter += 1
      paragraph_embeddings[i] += sentence
      paragraph_total += 1
  return paragraph_embeddings, sentence_embeddings, paragraph_lookup, sentence_lookup


paragraph_embeddings, sentence_embeddings, paragraph_lookup, sentence_lookup = get_embeddings(paras, embedding_size)
paragraph_embeddings = torch.tensor(np.array(paragraph_embeddings).astype(np.float)).type(torch.FloatTensor).to(device)
