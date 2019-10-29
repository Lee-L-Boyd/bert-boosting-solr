import pickle
import sys
from string import punctuation
import unicodedata
sys.path.append('../../IR_model/hybrid-BM25')
from squad_objects2 import *
from helper_functions import *


#converted sentences and saved them first 
all=pickle.load(open('../All_paras_tokenized_list.pickle', 'rb'))
print([len(a) for a in all[0].sentence_tokens])

new_all = [[clean_text(a)[1:] if len(clean_text(a)) > 0 and clean_text(a)[0] == ' ' else clean_text(a) for a in al.sentence_tokens] for al in all]

pickle.dump(new_all, open('../clean_para_id_plus_tok.pickle', 'wb'))



#got tfs and idfs and wrote to file then commented this for testing

global token_to_para
global para_token_count

token_to_para={}
para_token_count={}
all=pickle.load(open('../clean_para_id_plus_tok.pickle', 'rb'))
print([len(a) for a in all[0]])

def make_tables(a, i):
  try:
    token_to_para[a].add(i)
  except:
    token_to_para[a] = set()
    token_to_para[a].add(i)
  try:
    para_token_count[i][a]+=1
  except:
    try: 
      para_token_count[i]
    except:
      para_token_count[i] = {}
    para_token_count[i][a]=1

[[[make_tables(aa,i) for aa in a.split()] for a in al] for i, al in enumerate(all)]

idf = {}
for key in token_to_para.keys():
  idf[key]=len(token_to_para[key])


pickle.dump(idf, open('../clean_idfs.pickle', 'wb'))
pickle.dump(para_token_count, open('../clean_tfs.pickle', 'wb'))


test = pickle.load(open('../clean_tfs.pickle', 'rb'))
print(test[20238]['the'])
