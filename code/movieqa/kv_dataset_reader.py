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


def get_maxlen(*paths):
  maxlen = defaultdict(int)
  for path in paths:
    with open(path, 'r') as examples_file:
      fields = ['question', 'qn_entities', 'ans_entities', 'sources', 'relations', 'targets']
      reader = csv.DictReader(examples_file, delimiter=TAB, fieldnames=fields)
      for row in reader:
        example = {}
        example['question'] = row['question'].split(SPACE)
        example['qn_entities'] = row['qn_entities'].split(PIPE)
        example['ans_entities'] = row['ans_entities'].split(PIPE)
        example['sources'] = row['sources'].split(PIPE)
        example['relations'] = row['relations'].split(PIPE)
        example['targets'] = row['targets'].split(PIPE)

        maxlen['question'] = max(len(example['question']), maxlen['question'])
        maxlen['qn_entities'] = max(len(example['qn_entities']), maxlen['qn_entities'])
        maxlen['ans_entities'] = max(len(example['ans_entities']), maxlen['ans_entities'])
        maxlen['sources'] = max(len(example['sources']), maxlen['sources'])
        maxlen['relations'] = maxlen['sources']
        maxlen['targets'] = maxlen['sources']
  return maxlen


class DatasetReader(object):
  def __init__(self, args, maxlen, share_idx=True):
    self.share_idx = share_idx
    word_idx = read_file_as_dict(args.word_idx)
    self.word_idx_size = len(word_idx)
    entity_idx = read_file_as_dict(args.entity_idx)
    self.entity_idx_size = len(entity_idx)
    relation_idx = read_file_as_dict(args.relation_idx)
    self.relation_idx_size = len(relation_idx)
    idx = read_file_as_dict(args.idx)
    self.idx_size = len(idx)
    fields = ['question', 'qn_entities', 'ans_entities', 'sources', 'relations', 'targets']
    with open(args.input_examples, 'r') as input_examples_file:
      reader = csv.DictReader(input_examples_file, delimiter=TAB,
                              fieldnames=fields)
      self.maxlen = maxlen
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
        ##
        self.num_examples += 1
        examples.append(example)
      vec_examples = []
      for example in tqdm(examples):
        vec_example = {}
        for key in example.keys():
          encoder = None
          if key == 'question':
            encoder = word_idx
          elif key == 'relations':
            encoder = relation_idx
          else:
            encoder = entity_idx
          #override the dict to be used in encoding if dict has to be shared
          if self.share_idx:
            encoder = idx
          #answers are always encoded by entity_idx !!!
          if key == 'ans_entities':
            encoder = entity_idx
          vec_example[key] = [encoder[word] for word in example[key]]

          if key == 'ans_entities':
            # answers should be in [0,count_entities-1]!!!
            vec_example[key] = [label - 1 for label in vec_example[key]]

        vec_examples.append(vec_example)
    self.vec_examples = vec_examples

  def get_examples(self):
    return self.vec_examples

  def get_max_lengths(self):
    return self.maxlen

  def get_word_idx_size(self):
    return self.get_word_idx_size()

  def get_relation_idx_size(self):
    return self.relation_idx_size

  def get_entity_idx_size(self):
    return self.entity_idx_size

  def get_idx_size(self):
    return self.idx_size


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the kv file', required=True)
  parser.add_argument('--word_idx', help='word vocabulary', required=True)
  parser.add_argument('--entity_idx', help='entity vocabulary', required=True)
  parser.add_argument('--relation_idx', help='relation vocabulary', required=True)
  parser.add_argument('--idx', help='overall vocabulary', required=True)
  args = parser.parse_args()

  dr = DatasetReader(args)