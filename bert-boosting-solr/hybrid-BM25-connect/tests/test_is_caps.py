import pickle
import torch
queries = pickle.load(open('../../pickled_squad/training_queries_meta.pickle','rb'))
from squad_objects2 import *

caps_vector = []
quotes_vector = []
is_in_quotes = False
print(queries[0][0].question)

def get_caps_and_quotes(question):
  is_in_quotes = False
  caps_vector = []
  quotes_vector = []
  for index, s in enumerate(question.split()):
    if True:
      if s[0].isupper() and index !=0 and len(s) > 3:
        print("Is caps")
        caps_vector.append(1.)
      else:
        caps_vector.append(0.)
      if s[-1]=="\"":
        is_in_quotes = False
        print("Is in quotes")
        quotes_vector.append(1.)
      elif s[0]=="\"" or is_in_quotes:
        is_in_quotes = True
        print("is in quotes")
        quotes_vector.append(1.)
      else:
        quotes_vector.append(0.)
  return list(zip(caps_vector, quotes_vector))


for i, query in enumerate(queries[0][:50]):
  print(query.question)
  caps_vector = []
  quotes_vector = []
  for index, s in enumerate(query.question.split()):
    if True:
      if s[0].isupper() and index !=0 and len(s) > 3:
        print("Is caps")
        caps_vector.append(1)
      else:
        caps_vector.append(0)
      if s[-1]=="\"":
        is_in_quotes = False
        print("Is in quotes")
        quotes_vector.append(1)
      elif s[0]=="\"" or is_in_quotes:
        is_in_quotes = True
        print("is in quotes")
        quotes_vector.append(1)
      else:
        quotes_vector.append(0)
      print([cq for cq in get_caps_and_quotes(query.question)])
  print(caps_vector)
  print(quotes_vector)
  '''except:
    print("trouble processing heuristics")
'''
