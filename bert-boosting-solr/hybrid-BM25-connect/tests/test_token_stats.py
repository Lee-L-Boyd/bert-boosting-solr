import pickle
from squad_objects import *
from TokenStats import *
p=pickle.load(open('../../pickled_squad/token_stats.pickle','rb'))
print(p.get_token_f1('what'))

#p.metatexts = None
#pickle.dump(p, open('../../pickled_squad/token_stats.pickle','wb'))

