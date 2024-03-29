
from bert_serving.client import BertClient
bc = BertClient()
#example = bc.encode(['the cat fjjddfdfarted judgementally', 'this is it'], show_tokens=True)
#print(example)
import re
test1 = ['Following', 'the', 'disbandment', 'of', "Destiny's", 'Child', 'in', 'June', '2005,', 'she', 'released', 'her', 'second', 'solo', 'album,', "B'Day", '(2006),', 'which', 'contained', 'hits', '"Déjà', 'Vu",', '"Irreplaceable",', 'and', '"Beautiful', 'Liar".', 'Beyoncé', 'also', 'ventured', 'into', 'acting,', 'with', 'a', 'Golden', 'Globe-nominated', 'performance', 'in', 'Dreamgirls', '(2006),', 'and', 'starring', 'roles', 'in', 'The', 'Pink', 'Panther', '(2006)', 'and', 'Obsessed', '(2009).', 'Her', 'marriage', 'to', 'rapper', 'Jay', 'Z', 'and', 'portrayal', 'of', 'Etta', 'James', 'in', 'Cadillac', 'Records', '(2008)', 'influenced', 'her', 'third', 'album,', 'I', 'Am...', 'Sasha', 'Fierce', '(2008),', 'which', 'saw', 'the', 'birth', 'of', 'her', 'alter-ego', 'Sasha', 'Fierce', 'and', 'earned', 'a', 'record-setting', 'six', 'Grammy', 'Awards', 'in', '2010,', 'including', 'Song', 'of', 'the', 'Year', 'for', '"Single', 'Ladies', '(Put', 'a', 'Ring', 'on', 'It)".', 'Beyoncé', 'took', 'a', 'hiatus', 'from', 'music', 'in', '2010', 'and', 'took', 'over', 'management', 'of', 'her', 'career;', 'her', 'fourth', 'album', '4', '(2011)', 'was', 'subsequently', 'mellower', 'in', 'tone,', 'exploring', '1970s', 'funk,', '1980s', 'pop,', 'and', '1990s', 'soul.', 'Her', 'critically', 'acclaimed', 'fifth', 'studio', 'album,', 'Beyoncé', '(2013),', 'was', 'distinguished', 'from', 'previous', 'releases', 'by', 'its', 'experimental', 'production', 'and', 'exploration', 'of', 'darker', 'themes.']
test2 = ['Following', 'the', 'di', '##sband', '##ment', 'of', 'Destiny', "'", 's', 'Child', 'in', 'June', '2005', ',', 'she', 'released', 'her', 'second', 'solo', 'album', ',', 'B', "'", 'Day', '(', '2006', ')', ',', 'which', 'contained', 'hits', '"', 'D', '##é', '##j', '##à', 'V', '##u', '"', ',', '"', 'I', '##rre', '##place', '##able', '"', ',', 'and', '"', 'Beautiful', 'Lia', '##r', '"', '.', 'Beyoncé', 'also', 'ventured', 'into', 'acting', ',', 'with', 'a', 'Golden', 'Globe', '-', 'nominated', 'performance', 'in', 'Dream', '##girl', '##s', '(', '2006', ')', ',', 'and', 'starring', 'roles', 'in', 'The', 'Pink', 'Panther', '(', '2006', ')', 'and', 'O', '##bs', '##essed', '(', '2009', ')', '.', 'Her', 'marriage', 'to', 'rapper', 'Jay', 'Z', 'and', 'portrayal', 'of', 'E', '##tta', 'James', 'in', 'Cadillac', 'Records', '(', '2008', ')', 'influenced', 'her', 'third', 'album', ',', 'I', 'Am', '.', '.', '.', 'Sasha', 'Fi', '##er', '##ce', '(', '2008', ')', ',', 'which', 'saw', 'the', 'birth', 'of', 'her', 'alter', '-', 'ego', 'Sasha', 'Fi', '##er', '##ce', 'and', 'earned', 'a', 'record', '-', 'setting', 'six', 'Grammy', 'Awards', 'in', '2010', ',', 'including', 'Song', 'of', 'the', 'Year', 'for', '"', 'Single', 'Ladies', '(', 'Put', 'a', 'Ring', 'on', 'It', ')', '"', '.', 'Beyoncé', 'took', 'a', 'hiatus', 'from', 'music', 'in', '2010', 'and', 'took', 'over', 'management', 'of', 'her', 'career', ';', 'her', 'fourth', 'album', '4', '(', '2011', ')', 'was', 'subsequently', 'me', '##llow', '##er', 'in', 'tone', ',', 'exploring', '1970s', 'funk', ',', '1980s', 'pop', ',', 'and', '1990s', 'soul', '.', 'Her', 'critically', 'acclaimed', 'fifth', 'studio', 'album', ',', 'Beyoncé', '(', '2013', ')', ',', 'was', 'distinguished', 'from', 'previous', 'releases', 'by', 'its', 'experimental', 'production', 'and', 'exploration', 'of', 'darker', 'themes', '.']
test2_pointer = 0
hash = []
print(test2)
test2 = [re.sub('^##','',t) for t in test2]


for index, token in enumerate(test1):
  hash.append([])
  #print(token)
  token_combo = test2[test2_pointer]
  hash[index].append(test2_pointer)
  while token != token_combo and (test2_pointer+1)<len(test2) :
    test2_pointer+=1
    hash[index].append(test2_pointer)
    #print(test2[test2_pointer])
    token_combo += test2[test2_pointer]
  test2_pointer+=1

print(hash)
