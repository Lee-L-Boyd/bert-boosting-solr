def testList(the_list):
  return [a+5 for a in the_list]

a = [1,2,3]
b,c,d = testList(a)
print(b)
print(c)
print(d)
