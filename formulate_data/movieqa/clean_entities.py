#!/usr/bin/python

"""
Cleans up the entities file.
 - Removes commas
 - Deduplication of entities
 - Converts everything to lowercase
 - Sorts entities lexicographically
"""

import argparse

from sortedcontainers import SortedSet
from text_util import clean_word

def main(args):
  NEWLINE = "\n"
  entities_set = SortedSet([])
  count_raw = 0
  count_processed = 0
  with open(args.input, 'r') as entities_file:
    with open(args.output, 'w') as clean_entities_file:
      for entity in entities_file:
        count_raw += 1
        entity = clean_word(entity)
        if len(entity) > 0:
          entities_set.add(entity)
      for entity in entities_set:
        count_processed += 1
        clean_entities_file.write(entity + NEWLINE)
  print "COUNT_RAW: ", count_raw
  print "COUNT_PROCESSED: ", count_processed


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input', help='the raw entities.txt file', required=True)
  parser.add_argument('--output', help='the processed clean_entities.txt file', required=True)
  args = parser.parse_args()
  main(args)