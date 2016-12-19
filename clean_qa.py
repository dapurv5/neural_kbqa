#!/usr/bin/python

"""
Cleans up the qa training/test examples.
"""

import argparse
import csv

from text_util import clean_line
from text_util import clean_word
from clean_utils import read_file_as_set

def main(args):
  valid_entities_set = read_file_as_set(args.input_entities)
  with open(args.input_examples, 'r') as examples_file:
    with open(args.output_examples, 'w') as output_file:
      writer = csv.DictWriter(output_file, delimiter='\t', fieldnames=['question', 'answer'])
      for line in examples_file:
        line = clean_line(line)
        q, ans = line.split("?\t")
        q_words = q.split(" ")
        q_words = q_words[1:]
        q_words = [clean_word(w) for w in q_words]  # For Eg like (True Romance, when was it released?)
        ans_entities = ans.split(",")
        ans_entities = [clean_word(ans_entity) for ans_entity in ans_entities]

        valid_ans_entities = []
        has_invalid_word = False
        for word in ans_entities:
          if word in valid_entities_set:
            valid_ans_entities.append(word)
          else:
            has_invalid_word = True

        if has_invalid_word:
          ans = ans.replace(",", "")
          for entity in valid_entities_set:
            if ans.find(entity) > -1:
              valid_ans_entities.append(entity)

        writer.writerow({'question': ' '.join(q_words), 'answer': '|'.join(ans_entities)})



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_entities', help='the entities file', required=True)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  args = parser.parse_args()
  main(args)
