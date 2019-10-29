import pickle
import numpy as np
from squad_objects import *
p = pickle.load(open('All_para_meta.pickle', 'rb'))
print(np.argmax(p.calculate_score(Metatext(0, 'When did Beyonce record best-selling hit, Baby boy ?', 0))))
