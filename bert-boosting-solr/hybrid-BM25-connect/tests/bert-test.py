from bert_serving.client import BertClient
bc = BertClient()
example = bc.encode(['the cat farted judgementally', 'this is it'], show_tokens=True)
print(example[0][0][2])
print(example[0][1][2])
print(example[1][0])
print(example[1][1])


