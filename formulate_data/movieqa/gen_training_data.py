#!/usr/bin/python

import argparse
import csv
import random

from text_util import clean_word
from clean_utils import read_file_as_dict
from knowledge_graph import KnowledgeGraph
from search_index import SearchIndex
from question_parser import QuestionParser
from conf.conf import *

from tqdm import tqdm

PIPE = "|"
COMMA = ","





def main(args):
  pass



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_graph', help='the graph file', required=True)
  parser.add_argument('--input_doc', help='the doc file', required=False)
  parser.add_argument('--stopwords', help='stopwords file', required=False)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  parser.add_argument('--mode', help='train or test mode', required=True)
  args = parser.parse_args()

  knowledge_base = KnowledgeGraph(args.input_graph, unidirectional=False)
  main(args)