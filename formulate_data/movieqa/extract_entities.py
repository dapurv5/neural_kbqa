#!/usr/bin/python

import argparse
import csv
import random

from text_util import clean_word
from clean_utils import read_file_as_dict
from knowledge_graph import KnowledgeGraph
from search_index import SearchIndex
from question_parser import QuestionParser
from conf import *

from tqdm import tqdm

PIPE = "|"
COMMA = ","

class EntityExtractor(object):
  def __init__(self, input_graph, input_doc, stopwords):
    self.kb = KnowledgeGraph(input_graph, unidirectional=False)
    self.index = SearchIndex(input_doc)
    valid_entities_set = self.kb.get_entities()
    stop_vocab = read_file_as_dict(stopwords)
    self.qp = QuestionParser(valid_entities_set, stop_vocab)

  def get_question_entities(self, question):
    return self.qp.get_question_entities(question)

  def get_relevant_entities_from_index(self, question):
    return self.index.get_candidate_docs(question)

  def get_neighboring_entities(self, entities, num_hops=2):
    nbr_entities = set([])
    for entity in entities:
      for nbr in self.kb.get_candidate_neighbors(entity, num_hops=num_hops, avoid_high_degree_nodes=True):
        nbr_entities.add(nbr)
    return nbr_entities

  def remove_high_degree_ans_entities(self, ans_entities):
    """
    For questions which have more than one entity, try to remove high degree entities
    If everything is a high degree entity, randomly keep one of them.
    """
    if len(ans_entities) > 1 and REMOVE_HIGH_DEGREE_ANSWER_ENTITIES:
      ans_entities_clean = set([])
      for ans_entity in ans_entities:
        if ans_entity in self.kb.get_high_degree_entities():
          continue
        ans_entities_clean.add(ans_entity)
      if len(ans_entities_clean) > 0:
        ans_entities = ans_entities_clean
      else:
        ans_entities = random.sample(ans_entities, 1)
    return ans_entities

  def get_paths(self, qn_entity, ans_entity):
    return self.kb.get_all_paths(qn_entity, ans_entity, cutoff=MAX_PATH_LENGTH)



def main(args):
  ee = EntityExtractor(args.input_graph, args.input_doc, args.stopwords)
  with open(args.input_examples, 'r') as input_examples_file:
    with open(args.output_examples, 'w') as output_examples_file:
      reader = csv.DictReader(input_examples_file, delimiter='\t', fieldnames=['question', 'answer'])
      writer = csv.DictWriter(output_examples_file, delimiter='\t',
            fieldnames=['question', 'qn_entities', 'ans_entities', 'relevant_entities',
                        'nbr_qn_entities', 'nbr_relevant_entities',
                        'paths_of_entities', 'paths_of_relations'])
      writer.writeheader()
      max_count_candidate_entities = 0
      max_count_candidate_paths = 0
      for row in reader:
        qn_entities = ee.get_question_entities(row['question'])
        ans_str = row['answer']
        ans_entities = ans_str.split(PIPE)
        relevant_entities = ee.get_relevant_entities_from_index(row['question'])
        nbr_qn_entities = ee.get_neighboring_entities(qn_entities, num_hops=HOPS_FROM_QN_ENTITY)
        nbr_relevant_entities = ee.get_neighboring_entities(relevant_entities, num_hops=HOPS_FROM_RELEVANT_ENTITY)

        candidate_entities = set([])
        candidate_entities = candidate_entities.union(qn_entities)
        candidate_entities = candidate_entities.union(relevant_entities)
        candidate_entities = candidate_entities.union(nbr_qn_entities)
        candidate_entities = candidate_entities.union(nbr_relevant_entities)
        if args.mode == "train":
          candidate_entities = candidate_entities.union(ans_entities)

        #Remove high degree ans_entities
        ans_entities_clean = ee.remove_high_degree_ans_entities(ans_entities)

        max_count_candidate_entities = max(max_count_candidate_entities, len(candidate_entities))
        #Clip candidate entities by stochastically sampling a subset
        if len(candidate_entities) > MAX_CANDIDATE_ENTITIES:
          candidate_entities = random.sample(candidate_entities, MAX_CANDIDATE_ENTITIES)

        all_paths_of_entities = []
        all_paths_of_relations = []
        for ans_entity in ans_entities:
          ans_entity = clean_word(ans_entity)
          for qn_entity in qn_entities:
            paths_of_entities, paths_of_relations = ee.get_paths(qn_entity, ans_entity)
            all_paths_of_entities.extend(paths_of_entities)
            all_paths_of_relations.extend(paths_of_relations)

        max_count_candidate_paths = max(max_count_candidate_paths, len(all_paths_of_entities))
        #Clip candidate paths by stochastically sampling a subset of them
        if len(all_paths_of_entities) > MAX_CANDIDATE_PATHS:
          sampled_idx = random.sample(range(0, len(all_paths_of_entities)), MAX_CANDIDATE_PATHS)
          sampled_paths_of_entities = []
          sampled_paths_of_relations = []
          for idx in sampled_idx:
            sampled_paths_of_entities.append(all_paths_of_entities[idx])
            sampled_paths_of_relations.append(all_paths_of_relations[idx])
          all_paths_of_entities = sampled_paths_of_entities
          all_paths_of_relations = sampled_paths_of_relations

        output_row = {
          'question': row['question'],
          'qn_entities': get_str_of_seq(qn_entities),
          'ans_entities': get_str_of_seq(ans_entities),
          'relevant_entities': get_str_of_seq(relevant_entities),
          'nbr_qn_entities': get_str_of_seq(nbr_qn_entities),
          'nbr_relevant_entities': get_str_of_seq(nbr_relevant_entities),
          'paths_of_entities': get_str_of_nested_seq(all_paths_of_entities),
          'paths_of_relations': get_str_of_nested_seq(all_paths_of_relations)
        }
        writer.writerow(output_row)


    print "MAX COUNT CANDIDATE ENTITIES", max_count_candidate_entities
    print "MAX COUNT CANDIDATE PATHS", max_count_candidate_paths

def get_str_of_seq(entities):
  return PIPE.join(entities)

def get_str_of_nested_seq(paths):
  result = []
  for path in paths:
    result.append(COMMA.join(path))
  return PIPE.join(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_graph', help='the graph file', required=True)
  parser.add_argument('--input_doc', help='the doc file', required=False)
  parser.add_argument('--stopwords', help='stopwords file', required=False)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  parser.add_argument('--mode', help='train or test mode', required=True)
  args = parser.parse_args()
  main(args)