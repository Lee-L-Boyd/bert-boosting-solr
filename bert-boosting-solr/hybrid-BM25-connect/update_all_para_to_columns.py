import pickle
from squad_objects2 import *
import multiprocessing as mp
from multiprocessing import Pool
p=Pool(40)


#queries = pickle.load(open('../../pickled_squad/training_queries_meta.pickle','rb'))
all_paras = pickle.load(open('../../pickled_squad/All_para_meta.pickle', 'rb'))
def dictionary_lookup(dic, token):
  try:
    return dic[token]
  except: 
    return -1


#columns = [all_paras.matrix[dictionary_lookup(all_paras.global_dictionary, q)] for q in queries[0][0].question.split() if dictionary_lookup(all_paras.global_dictionary, q) != -1]
#for a in list(zip(queries[0][0].question.split(), columns)):
#  print(a[0])
#  print(a[1][0])
#columns = all_paras.matrix[dictionary_lookup(all_paras.global_dictionary, 'Taliban')]
#print(columns)
#print('Taliban' in all_paras.list_of_paras[0].dic.keys())
#all_paras.create_score_columns(1.2, .75, p)
#all_paras.create_score_matrix(1.2, .75, p)
all_paras.remake_score_columns(1.2, .75, p)
pickle.dump(all_paras, open('All_paras2.pickle', 'wb'))
all_paras = pickle.load(open('All_paras2.pickle', 'rb'))
for index, score in enumerate(all_paras.score_columns['Taliban']):
  if score > 0:
    print('Taliban' in all_paras.list_of_paras[index].dic.keys())

