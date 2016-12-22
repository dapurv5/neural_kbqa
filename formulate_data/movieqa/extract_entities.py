#!/usr/bin/python

import argparse
import csv

from clean_utils import read_file_as_dict
from knowledge_graph import KnowledgeGraph
from search_index import SearchIndex
from question_parser import QuestionParser

class EntityExtractor(object):
  def __init__(self, input_graph, input_doc, stopwords):
    self.kb = KnowledgeGraph(input_graph, unidirectional=False)
    self.index = SearchIndex(input_doc)
    valid_entities_set = self.kb.get_entities()
    stop_vocab = read_file_as_dict(stopwords)
    self.qp = QuestionParser(valid_entities_set, stop_vocab)

  def get_question_entities(self, question):
    return self.qp.get_question_entities(question)


def main(args):
  ee = EntityExtractor(args.input_graph, args.input_doc, args.stopwords)
  with open(args.input_examples, 'r') as input_examples_file:
    with open(args.output_examples, 'w') as output_examples_file:
      reader = csv.DictReader(input_examples_file, delimiter='\t', fieldnames=['question', 'answer'])
      writer = csv.DictWriter(output_examples_file, delimiter='\t',
            fieldnames=['question', 'qn_entities', 'ans_entities', 'candidate_entities', 'paths_of_entities', 'paths_of_relations'])
      for row in reader:
        qn_entities = ee.get_question_entities(row['question'])
        if len(qn_entities) != 1:
          print qn_entities, row['question']

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_graph', help='the graph file', required=True)
  parser.add_argument('--input_doc', help='the doc file', required=False)
  parser.add_argument('--stopwords', help='stopwords file', required=False)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  args = parser.parse_args()
  main(args)