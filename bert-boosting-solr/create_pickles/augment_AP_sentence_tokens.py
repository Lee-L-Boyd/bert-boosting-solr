import pickle
from squad_objects2 import *

paras_old = pickle.load(open('../para_encodings.pickle', 'rb'))
val_paras_old = pickle.load(open('../val_para_encodings.pickle', 'rb'))
all_old_paras = paras_old + val_paras_old
all_paras = pickle.load(open('../All_paras2.pickle', 'rb'))

all_paras.list_of_paras = all_paras.list_of_paras
all_paras.get_tokens_from_squad_paras(all_old_paras)
#print(all_paras.list_of_paras[0].sentence_tokens)


pickle.dump(all_paras.list_of_paras, open('../All_paras_tokenized_list.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
