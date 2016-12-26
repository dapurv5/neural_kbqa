#!/usr/bin/python

"""
Utilities for cleaning the text data
"""
import unicodedata

def clean_word(word):
  word = word.strip('\n')
  word = word.strip('\r')
  word = word.lower()
  word = word.replace('%', '') #99 and 44/100% dead
  word = word.strip()
  word = word.replace(',', '')
  word = word.replace('.', '')
  word = word.replace('"', '')
  word = word.replace('\'', '')
  word = word.replace('?', '')
  word = word.replace('|', '')
  word = unicode(word, "utf-8") #Convert str -> unicode (Remember default encoding is ascii in python)
  word = unicodedata.normalize('NFKD', word).encode('ascii','ignore') #Convert normalized unicode to python str
  word = word.lower() #Don't remove this line, lowercase after the unicode normalization
  return word


def clean_line(line):
  """
  Do not replace PIPE here.
  """
  line = line.strip('\n')
  line = line.strip('\r')
  line = line.strip()
  line = line.lower()
  return line

def append_word_to_str(text, str):
  if len(text) == 0:
    return str
  else:
    return text + " " + str

if __name__ == "__main__":
  print "__"+clean_word("  ")+"__"