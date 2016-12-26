#!/usr/bin/python

import csv

def read_file_as_set(input_path):
  s = set()
  with open(input_path) as input_file:
    for line in input_file:
      line = line.strip('\n')
      s.add(line)
  return s

def read_file_as_dict(input_path):
  d = {}
  with open(input_path) as input_file:
    reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
    for row in reader:
      d[row['col1']] = int(row['col2'])
  return d