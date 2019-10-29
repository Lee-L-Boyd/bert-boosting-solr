import pickle
import math
import urllib
import numpy as np
from squad_objects2 import *
import numpy as np

from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import re
import requests
from helper_functions import *
bc = BertClient()

total = 0.
points = 0.

p=pickle.load(open('../../pickled_squad/bad_results_training.pickle','rb'))
p2=pickle.load(open('../../pickled_squad/val_squad_questions.pickle','rb'))
p2 = [p for p in p2 if not p.is_impossible]
def check_if_bad_results(query, p):
  tester = True
  paragraphs = p
  for q in query:
    paragraphs = [paragraph for paragraph in p if q in paragraph[0][0].split()]
  try:
    paragraphs[0]
    return True
  except:
    return False

def get_answer(query, p):
  tester = True
  paragraphs = p
  for q in query:
    paragraphs = [paragraph for paragraph in paragraphs if q in paragraph.question.split()]
  try:
    #print(query)
    #print([para.question for para in paragraphs])
    return paragraphs[0].paragraph_num
  except:
    return -1

def get_positive_values(weights):
  min = 100000
  for a in weights:
    if a < min:
      min = a
  #print(weights)
  weights = [(a + -1* min) for a in weights ]
  #print(weights)
  return weights

def get_one_values(weights):
  return [(weight if weight >1 else 1) for weight in weights]

def get_zero_values(weights):
  return [(weight if weight >0 else 0) for weight in weights]

def choose(a, mean, std):
  if a < mean - 2 * std:
    return 1.
  elif a < mean - std:
    return 1.
  elif a <= mean + std:
    return 1.
  elif a <= mean + 2 * std:
    return 1.
  else:
    return 2.
def get_stats(weights):
  weights = np.array(weights)
  avg = np.average(weights)
  std = np.std(weights)
  return [choose(a, avg, std) for a in weights]
def query_db(style, string, weights, query_pre, answer, n_tries):
  #print("IN QUERY")
  is_in_quotes = False
  if style == 0:
    #bm25 unboosted
    content = ' '.join(string)
    boost_string = ['^'+str(int(round(1))) for a in weights]
    #print(boost_string)
    content = ' '.join([''.join(a) for a in zip(string, boost_string)])
    #print("IN zERO " + str(content))
  elif style == 1:
    #our approach positive values
    #weights = get_one_values(weights)
    weights = get_stats(weights)
    boost_string = ['^'+str(int(round(a*100))) for a in weights]
    #print(boost_string)
    content = ' '.join([''.join(a) for a in zip(string, boost_string)])
  elif style == 3:
    #our approach zero values
    weights = get_zero_values(weights)
    boost_string = ['^'+str(int(round(a*10))) for a in weights]
    #print(boost_string)
    content = ' '.join([''.join(a) for a in zip(string, boost_string)])

  elif style == 2:
    #our approach plus some additional heuristics
    for index, s in enumerate(string):
      try:
        #weights = get_stats(weights)
        #weights = get_zero_values(weights)
        avg = np.average(np.array(weights))
        if s[0].isupper() and index !=0 and len(s) > 3:
          #weights[index] += avg/4
          weights[index] += 1
        #if index == len(string)-1:
        #  weights[index]+= avg/4
        if s[0]=="\"" or is_in_quotes:
          is_in_quotes = True
          #weights[index]+=avg/4
          weights[index] += 1
        if s[-1]=="\"":
          is_in_quotes = False
      except:
        print("trouble processing heuristics")
        return -1, ''
      weights = get_stats(weights)
      boost_string = ['^'+str(int(round(a*100))) for a in weights]
      content = ' '.join([''.join(a) for a in zip(string, boost_string)])

  query_post = {'q' : content, 'fl' : 'para_id,score', 'sow':'false', 'rows':'5'}
  query = urllib.parse.urlencode(query_post)
  query = query.encode('utf-8').decode('utf-8')
  query = query_pre+query
  #print("Test")
  try:
    #print("test2")
    #print(content)
    connection = urllib2.urlopen(query)
    paragraphs = eval(connection.read())
    for index, p in enumerate(paragraphs['response']['docs']):
      if index < n_tries:
        if p['para_id'][0] == answer:
          return 1., content
    return 0., content
  except:
    print("Problem with connection************************" + str(query))
    return -1., ''
#query_pre = 'http://localhost:8983/solr/core_paras/select?'
query_pre = 'http://localhost:8983/solr/core_parsed_paras/select?'
content = 'How did Hayek feel regarding income distribution?'
strings, weightss, answers = pickle.load(open('../../pickled_squad/test_weights.pickle', 'rb'))
#strings = [s[:-1]+[strip_punctuation(s[-1])] for s in strings]
'''for s in strings:
  print(s)
  ' '.join(filter_BERT_sentence(get_BERT_tokens(bc, s)))'''
strings = [filter_BERT_sentence(get_BERT_tokens(bc, ' '.join(s))) for s in strings]
total_approaches = 1
correct = [0.]*total_approaches
total = [0.]*total_approaches
stored = [0.]*total_approaches
overall_attempts = [0.]*total_approaches
got_right=0
total_attempts = 0
chances_correct = 0.
chances_total = 0.
content = {}
for i, string in enumerate(strings):
  for j in range(total_approaches):
    if i < 20:
      print(string)
      print(weightss[i][0])
    is_correct, content[j] = query_db(j, string, weightss[i][0].copy(), query_pre,  answers[i], 1)
    #print(is_correct)
    if is_correct != -1. and len(content[j]) > 3:
      correct[j] += is_correct
      total[j] += 1.
      stored[j] = is_correct
      overall_attempts[j] = 1.
    else:
      stored[j] = 1.
  if np.sum(stored) != total_approaches and np.sum(stored)!= 0 or total_approaches == 1:
    for j in range(total_approaches):
      if total[j] != 0:
        print("query " +str(j) +": "+ str(correct[j]/total[j]))
      else:
        print("total for method " + str(i) +" is equal to zero cannot compute percent")
      if stored[j] == 1:
        print(content[j] + ": CORRECT!")
      else:
        print(content[j] + ": INCORRECT ):")
  is_correct, content[total_approaches] = query_db(0, string, weightss[i][0].copy(), query_pre,  answers[i], total_approaches)
  if is_correct != -1:
    chances_correct += is_correct
    chances_total += 1
  got_right += np.max(stored)
  total_attempts += np.max(overall_attempts)
  if np.sum(stored) != total_approaches and np.sum(stored) != 0 or total_approaches == 1:
    print("got_right total: " + str(got_right/total_attempts))
    print("chances_base: " + str(chances_correct/chances_total))
    print('')

'''
#connection = urllib2.urlopen('http://easel3.fulgentcorp.com:8983/solr/core_test1/select?fl=paragraph_id,content&fq=is_query:T&rows=10000000&q=*:*&fq=is_testing:T')
connection = urllib2.urlopen('http://localhost:8983/solr/core_queries/select?fl=id,paragraph_id,text&fq=is_query:T&rows=1000000&q=*:*&fq=is_testing:T&fq=is_impossible:F')
response = eval(connection.read())
query_pre = 'http://localhost:8983/solr/core_paras/select?'
bad_results = []
for r in response['response']['docs']:
  #content = 'content:' + re.sub('\s', ' or ', r['content'][0])
  #content = 'content:' + r['content'][0]
  #print(r.keys())
  content = r['text'][0]
  
  query_post = {'q' : content, 'fl' : 'para_id,score', 'fq': 'is_query:F', 'sow':'false', 'rows':'10'}
  query = urllib.parse.urlencode(query_post)
  query = query.encode('utf-8').decode('utf-8')
  query = query_pre+query
  #print(query)
  try:
    connection2 = urllib2.urlopen(query)
    paragraphs = eval(connection2.read())
    sanity_check = 0
    for p in paragraphs['response']['docs']:
      if r['paragraph_id'][0] == p['para_id'][0]:
        points += 1
        sanity_check+=1
        break
    if sanity_check > 1:
      print("Sanity check failed")
    if sanity_check == 0:
      bad_results.append((r['text'], r['paragraph_id']))
    sanity_check = 0
    total+=1
  except Exception as e:
    print(str(e))
    print(str(query))
print(points/total)
pickle.dump(bad_results, open('bad_results.pickle','wb'))
'''
