import json
import pickle
import re
import numpy as np 

from bert_serving.client import BertClient
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

import re

def generate_ngrams(tokens, n):
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

#0 for paras only, 1 for questions only, 2 for both
make_paras_or_questions = 2
#number_of_training_paras = 19034 + 1
if make_paras_or_questions != 1:
  bc = BertClient()
else:
  bc = None

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s

class squad_para(object):
    def __init__(self, para_id, tokens):
      self.para_id = para_id
      self.para_tokens = tokens
      self.para = ' '.join(tokens)
    def add_embeddings(self, bc):
      sentence_tokens = generate_ngrams(self.para_tokens, 10)
      #print(sentence_tokens)


      if bc is not None:
        #self.embedding_list = (bc.encode(combined), combined)
        #print(self.embedding_list)
        embedding_list = bc.encode(sentence_tokens)
      else:
        print("bc not defined")
      self.embedding_list = np.mean(np.array(embedding_list), axis=0)
      #print(embedding_list)
      #print(self.embedding_list)
class squad_question(object):
   def __init__(self, id, is_impossible, question, paragraph_num, answers, start, end):
     self.id = id
     self.question = question
     self.paragraph_num = paragraph_num
     self.is_impossible = is_impossible
     self.start = start
     self.end = end
     self.answers = answers

  
def read_squad_examples(input_file, is_training, version_2_with_negative, make_paras_or_questions, index):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    questions = []
    #index = -1
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            index+=1
            print(index)
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            if make_paras_or_questions != 1:
              para = squad_para(index, doc_tokens)
              para.add_embeddings(bc)
              examples.append(para)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    #if (len(qa["answers"]) != 1) and (not is_impossible):
                    #    raise ValueError(
                    #        "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                if make_paras_or_questions == 1 or make_paras_or_questions==2:
                  question_example = squad_question(qas_id,
                    is_impossible,
                    question_text,
                    index,
                    orig_answer_text,
                    start_position,
                    end_position)
                  questions.append(question_example)
    return (examples, questions, index)

import pickle
import re
max_len = 0
para_encodings = {}
last_doc_token = ''
counter = 0
(para_encodings, questions, index) = read_squad_examples('../../jsoned_squad/train-v2.0.json', True, True, make_paras_or_questions, -1)

if make_paras_or_questions == 0:
  pickle.dump(para_encodings, open('../para_conv.pickle','wb'))
elif make_paras_or_questions==1:
  pickle.dump(questions, open('../questions_conv.pickle','wb'))
else:
  pickle.dump(para_encodings, open('../para_conv.pickle','wb'))
  pickle.dump(questions, open('../questions_conv.pickle','wb'))

(para_encodings, questions, index) = read_squad_examples('../../jsoned_squad/dev-v2.0.json', True, True, make_paras_or_questions, index)

if make_paras_or_questions == 0:
  pickle.dump(para_encodings, open('../val_para_conv.pickle','wb'))
elif make_paras_or_questions==1:
  pickle.dump(questions, open('../val_questions_conv.pickle','wb'))
else:
  pickle.dump(para_encodings, open('../val_para_conv.pickle','wb'))
  pickle.dump(questions, open('../val_questions_conv.pickle','wb'))




#paras = [' '.join(QA_object.doc_tokens) for QA_object in read_squad_examples('train-v2.0.json', False, True)]
#print(bc.encode(paras))
#print(bc.encode(['this this this this']))
