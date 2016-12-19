#!/usr/bin/python

"""
Utilities for cleaning the text data
"""


def clean_word(word):
  word = word.strip('\n')
  word = word.lower()
  word = word.replace('%', '') #99 and 44/100% dead
  word = word.strip()
  word = word.replace(',', '')
  word = word.replace('.', '')
  word = word.replace('"', '')
  word = word.replace('\'', '')
  word = word.replace('?', '')
  return word


def clean_line(line):
  line = line.strip('\n')
  line = line.strip()
  line = line.lower()
  return line

def append_word_to_str(text, str):
  if len(text) == 0:
    return str
  else:
    return text + " " + str