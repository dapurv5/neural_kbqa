#!/usr/bin/python

import argparse
import csv
import sys
import random

import numpy as np

from clean_utils import read_file_as_dict
from data_utils import *

from collections import defaultdict
from tqdm import tqdm


class DatasetReader(object):
  def __init__(self, args):
    word_idx = read_file_as_dict(args.word_idx)
    entity_idx = read_file_as_dict(args.entity_idx)
    relation_idx = read_file_as_dict(args.relation_idx)
    fields = ['question', 'qn_entities', 'ans_entities', 'sources', 'relations', 'targets']
    with open(args.input_examples, 'r') as input_examples_file:
      reader = csv.DictReader(input_examples_file, delimiter=TAB,
                              fieldnames=fields)
      self.maxlen = defaultdict(int)
      self.num_examples = 0
      examples = []
      for row in tqdm(reader):
        example = {}
        example['question'] = row['question'].split(SPACE)
        example['qn_entities'] = row['qn_entities'].split(PIPE)
        example['ans_entities'] = row['ans_entities'].split(PIPE)
        example['sources'] = row['sources'].split(PIPE)
        example['relations'] = row['relations'].split(PIPE)
        example['targets'] = row['targets'].split(PIPE)
        self.maxlen['question'] = max(len(example['question']), self.maxlen['question'])
        self.maxlen['qn_entities'] = max(len(example['qn_entities']), self.maxlen['qn_entities'])
        self.maxlen['ans_entities'] = max(len(example['ans_entities']), self.maxlen['ans_entities'])
        self.maxlen['sources'] = max(len(example['sources']), self.maxlen['sources'])
        self.num_examples += 1
        examples.append(example)

      vec_examples = []
      for example in tqdm(examples):
        vec_example = {}
        vec_example['question'] = [word_idx[word] for word in example['question']]
        vec_example['qn_entities'] = [entity_idx[entity] for entity in example['qn_entities']]
        vec_example['ans_entities'] = [entity_idx[entity] for entity in example['ans_entities']]
        vec_example['sources'] = [entity_idx[entity] for entity in example['sources']]
        vec_example['relations'] = [relation_idx[relation] for relation in example['relations']]
        vec_example['targets'] = [entity_idx[entity] for entity in example['targets']]

        for key in vec_example.keys():
          vec_example[key] = np.array(vec_example[key])
        vec_examples.append(vec_example)
    self.vec_examples = vec_examples

  def get_examples(self):
    return self.vec_examples

  def get_max_lengths(self):
    return self.maxlen


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the kv file', required=True)
  parser.add_argument('--word_idx', help='word vocabulary', required=True)
  parser.add_argument('--entity_idx', help='entity vocabulary', required=True)
  parser.add_argument('--relation_idx', help='relation vocabulary', required=True)
  args = parser.parse_args()

  dr = DatasetReader(args)