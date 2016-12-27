#!/usr/bin/python

import os
import csv
import random

from collections import defaultdict
from tqdm import tqdm

from data_utils import union

PIPE = "|"
COMMA = ","
SPACE = " "
TAB = "\t"
NEWLINE = "\n"

words = set([])
entities = set([])
relations = set([])
all = set([])


def add_entity(entity):
  entities.add(entity)
  for word in entity.split(SPACE):
    words.add(word)

def add_sentence(sentence):
  for word in sentence.split(SPACE):
    words.add(word)

def read_graph_file(graph_path):
  with open(graph_path, 'r') as graph_file:
    print "reading graph file ..."
    reader = csv.DictReader(graph_file, delimiter=PIPE, fieldnames=['e1', 'relation', 'e2'])
    for row in tqdm(reader):
      entity1, relation, entity2 = row['e1'], row['relation'], row['e2']
      add_entity(entity1)
      add_entity(entity2)
      relations.add(relation)
      relations.add("INV_"+relation)

def read_doc_file(doc_path):
  with open(doc_path, 'r') as doc_file:
    print "reading doc file ..."
    reader = csv.DictReader(doc_file, delimiter=PIPE, fieldnames=['e', 'relation', 'description'])
    for row in tqdm(reader):
      entity, relation, description = row['e'], row['relation'], row['description']
      add_entity(entity)
      relations.add(relation)
      relations.add("INV_"+relation)
      add_sentence(description)

def read_qa_file(qa_path):
  with open(qa_path, 'r') as qa_file:
    print "reading qa file ..."
    reader = csv.DictReader(qa_file, delimiter=TAB, fieldnames=['question', 'answer'])
    for row in tqdm(reader):
      q, a = row['question'], row['answer']
      add_sentence(q)
      for e in a.split(PIPE):
        add_entity(e)

def write_idx(idx_path, s):
  print "writing ", idx_path, " ..."
  ordered = sorted(s)
  id = 1
  with open(idx_path, 'w') as idx_file:
    writer = csv.DictWriter(idx_file, delimiter="\t", fieldnames=['x', 'count'])
    for x in ordered:
      writer.writerow({'x': x, 'count': id})
      id = id + 1

if __name__ == "__main__":
  path_prefix = "/home/dapurv5/MyCode/anahata/src/play/python/neural_kbqa/data/movieqa/"
  dataset_name = "wiki-entities"
  train_path = path_prefix + "clean_{name}_qa_train.txt".format(name=dataset_name)
  test_path = path_prefix + "clean_{name}_qa_test.txt".format(name=dataset_name)
  dev_path = path_prefix + "clean_{name}_qa_dev.txt".format(name=dataset_name)
  graph_path = path_prefix + "clean_{name}_kb_graph.txt".format(name=dataset_name)
  doc_path = path_prefix + "clean_{name}_kb_doc.txt".format(name=dataset_name)
  read_graph_file(graph_path)
  read_doc_file(doc_path)
  #read_qa_file(train_path)
  read_qa_file(test_path)
  read_qa_file(dev_path)
  write_idx(path_prefix + "{name}_word_idx.txt".format(name=dataset_name), words)
  write_idx(path_prefix + "{name}_relation_idx.txt".format(name=dataset_name), relations)
  write_idx(path_prefix + "{name}_entity_idx.txt".format(name=dataset_name), entities)
  all = union(words, entities, relations)
  write_idx(path_prefix + "{name}_idx.txt".format(name=dataset_name), all)
