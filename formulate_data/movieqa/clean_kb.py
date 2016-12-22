#!/usr/bin/python

"""
Cleans up the kb file. Produces graph and doc files.
"""
import argparse
import csv

from text_util import clean_line
from text_util import clean_word
from clean_utils import read_file_as_set


def main(args):
  relations_set = set([])
  entities_set = set([])
  valid_entities_set = read_file_as_set(args.input_entities)
  with open(args.input_kb, 'r') as kb_file:
    with open(args.output_graph, 'w') as output_graph_file:
      with open(args.output_doc, 'w') as output_doc_file:
        graph_writer = csv.DictWriter(output_graph_file, delimiter='|',
                                      fieldnames=['subject', 'relation', 'object'])
        doc_writer = csv.DictWriter(output_doc_file, delimiter='|',
                                    fieldnames=['entity', 'fieldname', 'content'])
        for line in kb_file:
          line = clean_line(line)
          if len(line) == 0:
            continue
          e1, e2s, r = None, None, None
          cur = []
          found_relation = False
          for word in line.split(" ")[1:]:
            if '_' in word and not found_relation:
              r = word
              relations_set.add(r)
              e1 = " ".join(cur)
              cur = []
              found_relation = True
            else:
              cur.append(word)
          e2s = " ".join(cur)
          e1 = clean_word(e1)
          if r=="has_plot":
            write_doc(e1, e2s, r, valid_entities_set, doc_writer)
          else:
            write_tuples(e1, e2s, r, valid_entities_set, graph_writer, entities_set)
  print "COUNT_ENTITIES", len(entities_set)
  print "COUNT_RELATIONS", len(relations_set)


def write_doc(e1, e2s, relation, valid_entities_set, doc_writer):
  dict = {'entity': e1, 'content': clean_word(e2s), 'fieldname': relation}
  if e1 in valid_entities_set:
    doc_writer.writerow(dict)


def write_tuples(e1, e2s, relation, valid_entities_set, graph_writer, entities_set):
  for e2 in e2s.split(","):
    e2 = clean_word(e2)
    entities_set.add(e1)
    entities_set.add(e2)
    if e1 in valid_entities_set and e2 in valid_entities_set:
      write_to_graph_file(graph_writer, (e1, relation, e2))


def write_to_graph_file(graph_writer, tuple):
  dict = {'subject': tuple[0], 'relation': tuple[1], 'object': tuple[2]}
  graph_writer.writerow(dict)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_kb', help='the raw kb file', required=True)
  parser.add_argument('--input_entities', help='the processed entities file', required=True)
  parser.add_argument('--output_graph', help='the processed graph file', required=True)
  parser.add_argument('--output_doc', help='the processed document file', required=True)
  args = parser.parse_args()
  main(args)
