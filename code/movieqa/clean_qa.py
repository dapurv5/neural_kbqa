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
            has_invalid_word = True #there's a comma in one of the ans_entities

        if has_invalid_word:
          is_a_valid_split, valid_ans_entities = get_valid_entities(ans_entities, valid_entities_set, 0)
          valid_ans_entities = list(reversed(valid_ans_entities))

        #lastly if the line was messed up and you couldn't find valid entities, pick as many you can find and leave the invalid ones
        if len(valid_ans_entities) == 0:
          for word in ans_entities:
            if word in valid_entities_set:
              valid_ans_entities.append(word)

        if len(valid_ans_entities) > 0:
          writer.writerow({'question': ' '.join(q_words), 'answer': '|'.join(valid_ans_entities)})


def get_valid_entities(potential_entities, dictionary, pos):
  if pos >= len(potential_entities):
    return True, []

  for i in range(pos, len(potential_entities)):
    chunk = " ".join(potential_entities[pos:i+1])
    if chunk in dictionary:
      is_a_valid_split, chunks = get_valid_entities(potential_entities, dictionary, i+1)
      if is_a_valid_split:
        chunks.append(chunk)
        return True, chunks
  return False, []


def test_get_valid_entities():
  print get_valid_entities(["monster in law", "they shoot horses", "don't they", "agnes of god"],
                           set(["monster in law", "they shoot horses don't they",
                                "agnes", "agnes of god", "a", "agnes"]), 0)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_entities', help='the entities file', required=True)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  args = parser.parse_args()
  main(args)
