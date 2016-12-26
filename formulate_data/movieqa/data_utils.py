#!/usr/bin/python

PIPE = "|"
COMMA = ","

def union(*sets):
  target_set = set([])
  for s in sets:
    target_set = target_set.union(s)
  return target_set


def extract_dimension_from_tuples_as_list(list_of_tuples, dim):
  result = []
  for tuple in list_of_tuples:
    result.append(tuple[dim])
  return result


def get_str_of_seq(entities):
  return PIPE.join(entities)


def get_str_of_nested_seq(paths):
  result = []
  for path in paths:
    result.append(COMMA.join(path))
  return PIPE.join(result)