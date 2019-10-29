import pickle 
from squad_objects2 import *
from multiprocessing import Pool
import random

if __name__=='__main__':
  p = Pool(40)


  training_queries = pickle.load(open('../questions_conv.pickle','rb'))
  training_queries = [q for q in training_queries if not q.is_impossible]
  print(len(training_queries))
  
  #validation_queries = pickle.load(open('../training_queries_meta.pickle','rb'))
  #print(len(validation_queries))
  #exit()

  num_validation_questions = int(len(training_queries)/10)
  num_training_questions = len(training_queries) - num_validation_questions
  mixed_up_training_questions = random.sample(training_queries, k=len(training_queries))

  validation_queries = mixed_up_training_questions[-num_validation_questions:]
  training_queries = mixed_up_training_questions[:num_training_questions]

  training_queries = p.starmap(Metatext, zip([0]*len(training_queries), [q.question for q in training_queries], [q.paragraph_num for q in training_queries]))
  validation_queries = p.starmap(Metatext, zip([0]*len(validation_queries), [q.question for q in validation_queries], [q.paragraph_num for q in validation_queries]))

  pickle.dump([training_queries], open('../training_queries_meta.pickle','wb'))
  
  pickle.dump([validation_queries], open('../val_training_queries_meta.pickle','wb'))
  print(len(training_queries))

  testing_queries = pickle.load(open('../../pickled_squad/val_questions_conv.pickle','rb'))
  testing_queries = [q for q in testing_queries if not q.is_impossible]
  testing_queries = p.starmap(Metatext, zip([0]*len(testing_queries), [q.question for q in testing_queries], [q.paragraph_num for q in testing_queries]))
  pickle.dump([testing_queries], open('../testing_queries_meta.pickle','wb'))
  print(len(testing_queries))
