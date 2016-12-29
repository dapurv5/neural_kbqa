#!/usr/bin/python

import argparse
import csv
import random

from text_util import clean_word
from clean_utils import read_file_as_dict
from knowledge_graph import KnowledgeGraph
from search_index import SearchIndex
from question_parser import QuestionParser
from data_utils import *

from tqdm import tqdm


MAX_RELEVANT_ENTITIES = 4
HOPS_FROM_QN_ENTITY = 1
MAX_CANDIDATE_ENTITIES = 64
MAX_CANDIDATE_TUPLES = 512


def remove_high_degree_qn_entities(qn_entities):
  """ Remove all high degree entities from the question except one"""
  qn_entities_clean = set([])
  if len(qn_entities) > 1:
    for qn_entity in qn_entities:
      if qn_entity not in knowledge_base.get_high_degree_entities():
        qn_entities_clean.add(qn_entity)
  return qn_entities_clean if len(qn_entities_clean) > 0 else qn_entities

def remove_invalid_ans_entities(ans_entities):
  ans_entities_clean = set([])
  for ans_entity in ans_entities:
    if ans_entity in knowledge_base.get_entities():
      ans_entities_clean.add(ans_entity)
  return ans_entities_clean if len(ans_entities_clean) > 0 else ans_entities


def get_neighboring_entities(entities, num_hops=2):
  nbr_entities = set([])
  for entity in entities:
    for nbr in knowledge_base.get_candidate_neighbors(entity, num_hops=num_hops,
                                                      avoid_high_degree_nodes=True):
      nbr_entities.add(nbr)
  return nbr_entities

def get_tuples_involving_entities(candidate_entities):
  tuples = set([])
  for s in candidate_entities:
    if s in knowledge_base.get_high_degree_entities():
      continue
    for t in knowledge_base.get_adjacent_entities(s):
      r = knowledge_base.get_relation(s,t)
      tuples.add((s, r, t))
  return tuples


def main(args):
  with open(args.input_examples, 'r') as input_examples_file:
    with open(args.output_examples, 'w') as output_examples_file:
      reader = csv.DictReader(input_examples_file, delimiter='\t', fieldnames=['question', 'answer'])
      writer = csv.DictWriter(output_examples_file, delimiter='\t',
                              fieldnames=['question', 'qn_entities', 'ans_entities',
                                          'sources', 'relations', 'targets'])
      for row in tqdm(reader):
        answer = row['answer']
        ans_entities = answer.split(PIPE)
        ans_entities = remove_invalid_ans_entities(ans_entities)
        question = row['question']
        qn_entities = question_parser.get_question_entities(question)
        qn_entities = remove_high_degree_qn_entities(qn_entities)
        relevant_entities = search_index.get_candidate_docs(question, limit=MAX_RELEVANT_ENTITIES)
        nbr_qn_entities = get_neighboring_entities(qn_entities, num_hops=HOPS_FROM_QN_ENTITY)
        candidate_entities = union(qn_entities, relevant_entities, nbr_qn_entities)
        # Clip candidate entities by stochastically sampling a subset
        if len(candidate_entities) > MAX_CANDIDATE_ENTITIES:
          candidate_entities = set(random.sample(candidate_entities, MAX_CANDIDATE_ENTITIES))
        tuples = get_tuples_involving_entities(candidate_entities)
        if len(tuples) > MAX_CANDIDATE_TUPLES:
          tuples = set(random.sample(tuples, MAX_CANDIDATE_TUPLES))
        sources = extract_dimension_from_tuples_as_list(tuples, 0)
        relations = extract_dimension_from_tuples_as_list(tuples, 1)
        targets = extract_dimension_from_tuples_as_list(tuples, 2)
        output_row = {
          'question': question,
          'qn_entities': get_str_of_seq(qn_entities),
          'ans_entities': get_str_of_seq(ans_entities),
          'sources': get_str_of_seq(sources),
          'relations': get_str_of_seq(relations),
          'targets': get_str_of_seq(targets)
        }
        writer.writerow(output_row)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_graph', help='the graph file', required=True)
  parser.add_argument('--input_doc', help='the doc file', required=False)
  parser.add_argument('--stopwords', help='stopwords file', required=False)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  args = parser.parse_args()

  #global variables
  knowledge_base = KnowledgeGraph(args.input_graph, unidirectional=False)
  search_index = SearchIndex(args.input_doc, args.stopwords)
  stop_vocab = read_file_as_dict(args.stopwords)
  question_parser = QuestionParser(knowledge_base.get_entities(), stop_vocab)
  main(args)