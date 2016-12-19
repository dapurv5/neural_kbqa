#!/usr/bin/python


def read_file_as_set(input_path):
  s = set()
  with open(input_path) as valid_entities_file:
    for entity in valid_entities_file:
      entity = entity.strip('\n')
      s.add(entity)
  return s