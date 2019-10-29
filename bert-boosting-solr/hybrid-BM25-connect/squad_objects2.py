import math 
import numpy as np
import multiprocessing as mp
import re
print("Number of processors: ", mp.cpu_count())
from string import punctuation

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

class squad_para(object):
    def __init__(self, para_id, tokens):
      self.para_id = para_id
      self.para_tokens = tokens
      self.length = len(tokens)
      self.para = ' '.join(tokens)
      self.dictionary = self.create_dictionary()
    def re_init(self):
      self.tokenize()
      self.length = len(self.para_tokens)
      self.dictionary = self.create_dictionary()
    def calculate_length(self):
      self.length = len(self.para_tokens)
    def tokenize_bert(self, bc):
      #combined = [s for s in self.para.split('.')]
      #semicolon_split = []
      #for c in combined:
      #  semicolon_split+=[s+' .' for s in c.split(';') if len(s) > 5]
      #combined = semicolon_split
      #sentence_encodings = []
      if bc is not None:
        tokens_and_embeddings = bc.encode([self.para], show_tokens = True)
        self.para_tokens = [t for t in tokens_and_embeddings[1][0] if t not in ['[CLS]', '[SEP]', '0_PAD']]
        #print(self.para_tokens)
      else:
        print("bc not defined")
    def tokenize(self):
      self.para_tokens = self.para.split()
    def create_dictionary(self):
      dictionary = {}
      for token in self.para_tokens:
        if token in dictionary.keys():
          dictionary[token] += 1
        else:
          dictionary[token] = 1
      return dictionary

class squad_question(object):
   def __init__(self, id, is_impossible, question, paragraph_num, answers, start, end, bc):
     self.id = id
     self.question = question
     self.tokens = self.tokenize(bc)
     self.paragraph_num = paragraph_num
     self.is_impossible = is_impossible
     self.start = start
     self.end = end
     self.answers = answers
     self.dictionary = self.create_dictionary()
   def re_init(self, bc):
     self.tokens = self.tokenize(bc)
     self.dictionary = self.create_dictionary()
   def parse(self):
     #print("ATTEMPTING PARSE")
     try:
       #print(self.question.rstrip().split())
       self.question = self.question.rstrip().split()
     except:
       print("could not parse (may already be parsed)")
   def tokenize(self, bc):
     self.tokens = self.question.split()

   def tokenize_bert(self, bc):
      if bc is not None:
        tokens_and_embeddings = bc.encode([self.question], show_tokens = True)
        self.tokens = [t for t in tokens_and_embeddings[1][0] if t not in ['[CLS]', '[SEP]', '0_PAD']]
      else:
        print("bc not defined")

   def create_dictionary(self):
     dictionary = {}
     for token in self.question.split():
       if token in dictionary.keys():
         dictionary[token] += 1
       else:
         dictionary[token] = 1
     return dictionary




class squad_paras(object):
  def __init__(self, list_of_paras,k1,b,bc):
    self.list_of_paras = list_of_paras
    try:
      self.num_docs = self.get_num_docs()
      self.global_dictionary = self.combine_all_dicts()
      self.avg_length = self.compute_avg_length()
      self.idfs = self.compute_idfs()
      self.word_to_id = self.create_word_to_id()
      self.matrix = self.create_score_matrix(k1,b)
      self.bc = bc
    except:
      print("not initialized")
      self.bc = bc
      self.re_init(k1, b)
  def re_init(self, k1, b):
    [p.re_init() for p in self.list_of_paras]
    self.num_docs = self.get_num_docs()
    self.global_dictionary = self.combine_all_dicts()
    self.avg_length = self.compute_avg_length()
    self.idfs = self.compute_idfs()
    self.word_to_id = self.create_word_to_id()
    self.matrix = self.create_score_matrix(k1,b)

  def create_word_to_id(self):
    temp_dictionary = {}
    for index, word in enumerate(self.global_dictionary.keys()):
      temp_dictionary[word] = index
  def get_num_docs(self):
    return len(self.list_of_paras)
  def combine_all_dicts(self):
    total_dictionary = {}
    for p in self.list_of_paras:
      p.create_dictionary()
      total_dictionary = combine_dictionaries_for_idf(total_dictionary, p.dictionary)
    return total_dictionary

  def compute_avg_length(self):
    total_length = 0.
    counter = 0.
    for p in self.list_of_paras:
      p.calculate_length()
      total_length += p.length
      counter += 1
    return total_length/counter
  def compute_idfs(self):
    temp_dict = {}
    num_docs = len(self.list_of_paras)
    for key in self.global_dictionary.keys():
       temp_dict[key] = math.log((num_docs - self.global_dictionary[key] + .5) / (self.global_dictionary[key] + .5))
    return temp_dict
  def create_score_matrix(self, k1, b):
    matrix = []
    for para in self.list_of_paras:
      doc_column = []
      for word in self.global_dictionary.keys():
        if word in para.dictionary.keys():
          frequency = para.dictionary[word]
          idf = self.idfs[word]
          nom = frequency * (k1 + 1.)
          denom = frequency + k1 * (1. - b + b * para.length/self.avg_length)
          cell_total = idf * nom / denom
          doc_column.append(cell_total)
        else:
          doc_column.append(0.)
      matrix.append(doc_column)
    return np.array(matrix).T
  def calculate_score(self, query):
    total = 0
    counts = []
    for word in self.global_dictionary.keys():
     try:
       counts.append(query.dictionary[word])
     except:
       counts.append(0)
    counts = np.array([counts])
    return np.argmax(np.squeeze(np.dot(counts,self.matrix)))
  
  def get_ids(self):
    return [p.para_id for p in self.list_of_paras]

def generate_dic(dic, n):
  num = 0 
  while num < n:
    yield dic
    num+=1


class Metatext(object):
  def __init__(self, size, dictionary, id):
    self.size = size
    if isinstance(dictionary, str):
      self.dic = self.f_queries(dictionary)
    else:
      self.dic = dictionary 
    self.id = id
  def add_tokens(self, paras):
    if True:
      para = [p.para for p in paras if p.para_id == self.id][0]
      combined = [s for s in para.split('.')]
      semicolon_split = []
      for c in combined:
        semicolon_split+=[s+'.' for s in c.split(';') if len(s) > 1]
      combined = semicolon_split
      print(combined)
      self.sentence_tokens = combined
    '''except:
      print("UNABLE TO FIND TOKENS FOR PARAGRAPH " + str(self.id))
    '''

  def f_queries(self, query):
   dictionary = {}
   self.question = query 
   self.tokens = query.split()
   for token in self.tokens:
     if token in dictionary.keys():
       dictionary[token] += 1
     else:
       dictionary[token] = 1
   return dictionary






  def get_embs_tokens(self, bc):
    #Note: this is just the sentence untokenized
    original_tokens = [re.sub('\?$', '', token) for token in self.tokens]
    self.tokens = original_tokens
    #original_tokens = [s[:-1]+[self.strip_punctuation(s[-1])] for s in original_tokens]
    #original_tokens[-1] = self.strip_punctuation(original_tokens[-1])
    bertized_query = bc.encode([self.question], show_tokens=True)
    tokens = [re.sub('^##|\?$','',e) for e in bertized_query[1][0] if e not in ['[CLS]', '[SEP]', '0_PAD']]
    embeddings = bertized_query[0]
    token_pointer = 0
    original_token_embeddings = []
    embeddings = np.squeeze(embeddings)
    #print(embeddings[2][0])
    #print(embeddings[3][0])
    #tokens = [re.sub('^##','',t) for t in tokens[0]]
    for index, o_token in enumerate(original_tokens):
      counter = 1
      embedding = np.array(embeddings[token_pointer])
      #print(tokens)
      token_combo = tokens[token_pointer]
      while o_token != token_combo and (token_pointer+1)< len(tokens):
        token_pointer += 1
        embedding += np.array(embeddings[token_pointer])
        token_combo += tokens[token_pointer]
        counter += 1
      token_pointer += 1
      embedding/=counter
      original_token_embeddings.append(embedding)
    return original_token_embeddings
  def get_para_embs(self, bc):
    return bc.encode([self.question])
  def strip_punctuation(self, s):
    return ''.join(c for c in s if c not in punctuation)


class All_para_meta(object):
  def __init__(self, stats, list_of_metas, k1, b, p):
    self.num_docs = len(list_of_metas)
    self.global_dictionary = stats.dic
    self.avg_length = stats.size*1./self.num_docs
    self.list_of_paras = list_of_metas
    self.idfs = self.compute_idfs()
    self.matrix = []
    self.score_columns = create_score_columns(k1, b, p)
  def get_tokens_from_squad_paras(self, squad_paras):
    [p.add_tokens(squad_paras) for p in self.list_of_paras]
  def remake_score_columns(self, k1, b, p):
    self.matrix = []
    self.score_columns = self.create_score_columns(k1, b, p)
  def create_score_columns(self, k1, b, p):
    matrix = []
    result_dict = {}
    score_columns = {}
    #list_of_paras = self.list_of_paras
    #global_dictionary = self.global_dictionary
    #num_docs = self.num_docs
    columns = p.starmap(self.make_column,
      zip(self.global_dictionary.keys(),
        generate_dic(self.list_of_paras, len(self.global_dictionary.keys())),
        generate_dic(self.idfs, len(self.global_dictionary.keys())),
        generate_dic(self.avg_length, len(self.global_dictionary.keys())),
        generate_dic(k1, len(self.global_dictionary.keys())),
        generate_dic(b, len(self.global_dictionary.keys()))))
    for tuple in columns:
      score_columns[tuple[0]]=tuple[1]
    return score_columns
  def compute_idfs(self):
    temp_dict = {}
    for key in self.global_dictionary.keys():
       temp_dict[key] = math.log((self.num_docs - self.global_dictionary[key] + .5) / (self.global_dictionary[key] + .5))
    return temp_dict
  def calculate_score(self, query):
    total = 0
    counts = []
    for word in self.global_dictionary.keys():
     try:
       counts.append(query.dic[word])
     except:
       counts.append(0)
    counts = np.array(counts)
    return np.dot(counts,self.matrix)
  def make_column(self, word, para_list, idfs, avg_length, k1, b):
    word_column = []
    for para in para_list:
      if word in para.dic.keys():
        frequency = para.dic[word]
        idf = idfs[word]
        nom = frequency * (k1 + 1.)
        denom = frequency + k1 * (1. - b + b * para.size/avg_length)
        cell_total = idf * nom / denom
        word_column.append(cell_total)
      else:
        word_column.append(0.)
    return (word, np.array(word_column))
  def get_column(self, word):
    try:
      return self.score_columns[word]
    except:
      print("unable to find token " +str(word) + " in score columns of all_paras")
      return [0.] * self.num_docs

  def get_columns(self, words):
    return [self.get_column(word) for word in words]
