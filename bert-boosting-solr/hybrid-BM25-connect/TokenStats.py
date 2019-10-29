import pickle
from multiprocessing import Pool
from squad_objects2 import *
import itertools
import numpy as np




class TokenStats():
  def __init__(self, queries_file, all_paras_file):
    #Note: old way to get queries to metas
    #self.queries = self.squad_questions_to_metas('training_', queries_file)
    self.queries = pickle.load(open(queries_file, 'rb'))[0]
    self.metatexts = pickle.load(open(all_paras_file, 'rb')).list_of_paras
    self.token_to_doc_lookup = {}
    self.build_inverse_table()
    self.token_stats = {}
    self.token_stats['THE AVERAGE TOKEN'] = np.array([0.,0.,0.])
    self.build_token_stats()

  def squad_questions_to_metas(self, pretense, file):
    p=Pool(40)
    queries = pickle.load(open(file,'rb'))
    queries_meta = p.starmap(Metatext, zip([len(q.question.split()) for q in queries], [q.question for q in queries], [q.paragraph_num for q in queries]))
    pickle.dump([queries_meta], open('../../pickled_squad/'+pretense+'queries_meta.pickle','wb'))
    return queries_meta

  def build_inverse_table(self):
    for metatext in self.metatexts:
      for token in metatext.dic.keys():
        try:
          self.token_to_doc_lookup[token].append(metatext.id)
        except:
          self.token_to_doc_lookup[token] = [metatext.id]

  def intersection( self, list1, list2):
    return list(set(list1) & set(list2))

  def build_token_stats(self):
    for query in self.queries:
      for token in query.dic.keys():
        if token in self.token_to_doc_lookup.keys():
          list_of_docs_containing_token = self.token_to_doc_lookup[token]
          #NOTE: I made query.id into a list to show that this could be used for a list of paragraphs, if there is not a one to one relation b/w para and query
          tp = len(self.intersection(list_of_docs_containing_token, [query.id])) * 1.
          fn_and_tp = len([query.id]) * 1.
          fp_and_tp = len(list_of_docs_containing_token) * 1.
          self.update_token_stats(token, tp, fn_and_tp, fp_and_tp)
        else:
          self.update_token_stats(token, 0., 1., 1.)

  def update_token_stats(self, token, tp, fn_and_tp, fp_and_tp):
    if token in self.token_stats.keys():
      self.token_stats[token] += np.array([tp, fn_and_tp, fp_and_tp])
    else:
      self.token_stats[token] = np.array([tp, fn_and_tp, fp_and_tp])
    self.token_stats['THE AVERAGE TOKEN'] += np.array([tp/10000, fn_and_tp/10000, fp_and_tp/10000])

  def get_token_precision(self, token):
    if token in self.token_stats.keys() and self.token_stats[token][0]/self.token_stats[token][2]>0.0:
      return self.token_stats[token][0]/self.token_stats[token][2]
    else:
      return self.token_stats['THE AVERAGE TOKEN'][0]/self.token_stats['THE AVERAGE TOKEN'][2]
  
  def get_token_recall(self, token):
    if token in self.token_stats.keys() and self.token_stats[token][0]/self.token_stats[token][1]>0.0:
      return self.token_stats[token][0]/self.token_stats[token][1]
    else:
      return self.token_stats['THE AVERAGE TOKEN'][0]/self.token_stats['THE AVERAGE TOKEN'][1]

  def get_token_f1(self, token):
    precision = self.get_token_precision(token)
    #print(precision)
    recall = self.get_token_recall(token)
    #print(recall)
    return 2 * precision * recall / (precision + recall) 


#old way loading from squad_object
#token_stats = TokenStats( '../../pickled_squad/squad_questions.pickle','../../pickled_squad/All_para_meta.pickle')
#token_stats.squad_questions_to_metas('val_', '../../pickled_squad/val_squad_questions.pickle')

#token_stats = TokenStats( '../../pickled_squad/training_queries_meta.pickle','../../pickled_squad/All_para_meta.pickle')


#pickle.dump(token_stats, open('../../pickled_squad/token_stats.pickle','wb'))
