from multiprocessing import Pool
from test_iterator_starmap_function import *
p = Pool(2)

if __name__ == '__main__':
  words = ['test', 'test2', 'test3']

  a = {}
  a['test'] = 1.
  a['test2'] = 2.

  b = {}
  b['test2'] = 4.
  b['test'] = 3.

  def generate_dic(dic, n):
    num = 0
    while num < n:
      yield dic
      num+=1

  columns = p.starmap(test2, zip( words, generate_dic(a, len(words)), generate_dic(b, len(words)) ) ) 
  print(columns)

