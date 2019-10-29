def test2(word, a, b):
  column = []
  if word in a.keys():
    column.append(a[word])
  else:
    column.append(0.)
  if word in b.keys():
    column.append(b[word])
  else:
    column.append(0.)
  return [word, column]

