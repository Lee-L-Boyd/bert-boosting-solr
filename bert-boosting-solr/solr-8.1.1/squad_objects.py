class squad_para(object):
    def __init__(self, para_id, tokens):
      self.para_id = para_id
      self.para_tokens = tokens
      self.para = ' '.join(tokens)
    def add_embeddings(self, bc):
      
      combined = [s for s in self.para.split('.')]
      semicolon_split = []
      for c in combined:
        semicolon_split+=[s+' .' for s in c.split(';') if len(s) > 5]
      combined = semicolon_split
      sentence_encodings = []
      if bc is not None:
        self.embedding_list = (bc.encode(combined), combined)
        print(self.embedding_list)
      else:
        print("bc not defined")
class squad_question(object):
   def __init__(self, id, is_impossible, question, paragraph_num, answers, start, end):
     self.id = id
     self.question = question
     self.paragraph_num = paragraph_num
     self.is_impossible = is_impossible
     self.start = start
     self.end = end
     self.answers = answers

