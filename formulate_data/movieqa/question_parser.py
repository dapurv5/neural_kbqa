#!/usr/bin/python

import argparse
import csv


class QuestionParser(object):
  def __init__(self, valid_entities_set, stop_vocab):
    self.valid_entities_set = valid_entities_set
    self.stop_vocab = stop_vocab

  def remove_all_stopwords_except_one(self, qn_entities):
    qn_entities_clean = set([])
    # Remove stop entities
    for entity in qn_entities:
      if entity not in self.stop_vocab:
        qn_entities_clean.add(entity)
    #If all entities were stopwords, keep the one with least freq.
    if len(qn_entities_clean) == 0:
      least_freq_stop_entity = qn_entities[0]
      for entity in qn_entities:
        if self.stop_vocab[entity] < self.stop_vocab[least_freq_stop_entity]:
          least_freq_stop_entity = entity
      qn_entities_clean.add(least_freq_stop_entity)
    return qn_entities_clean

  def remove_substrings(self, qn_entities):
    if len(qn_entities) > 1:
      qn_entities_clean = set([])
      #Remove entities contained in other entities
      for entity1 in qn_entities:
        is_substring_of = False
        for entity2 in qn_entities:
          if entity1 == entity2:
            continue
          if entity1 in entity2:
            is_substring_of = True
            break
        if not is_substring_of:
          qn_entities_clean.add(entity1)
      qn_entities = qn_entities_clean
    return list(qn_entities)

  def get_sets_after_difference(self, s1, s2):
    """ Returns difference s1-s2 and s2-s1 where minus sign indicates set difference"""
    intersection = s1.intersection(s2)
    for e in intersection:
      s1.remove(e)
      s2.remove(e)
    return s1, s2

  def get_sets_after_removing_stopwords(self, s1, s2):
    score1, score2 = 0, 0
    for word in list(s1):
      if word in self.stop_vocab:
        score1 = score1 + self.stop_vocab[word]
        s1.remove(word)
    for word in list(s2):
      if word in self.stop_vocab:
        score2 = score2 + self.stop_vocab[word]
        s2.remove(word)
    return s1, score1, s2, score2

  def remove_spurious_entities(self, qn_entities, question):
    #Remove spurious entities
    if len(qn_entities) > 1:
      qn_entities_clean = set([])
      for entity1 in qn_entities:
        for entity2 in qn_entities:
          if entity1 == entity2:
            continue
          s1, s2 = set(entity1.split(" ")), set(entity2.split(" "))
          pos1, pos2 = question.find(entity1), question.find(entity2)
          intersection = s1.intersection(s2)
          #If there is no intersection, none of them can be spurious and cannot be removed
          if len(intersection) == 0:
            qn_entities_clean.add(entity1)
          if pos1 < pos2 and pos1 + len(entity1) > pos2:
            #e1 lies to the left of e2 and e1 and e2 were picked from spatially intersecting windows
            s1, s2 = self.get_sets_after_difference(s1, s2)
            s1, score1, s2, score2 = self.get_sets_after_removing_stopwords(s1, s2)

            #Case 1: e1 is not spurious
            if len(s1) > 0:
              qn_entities_clean.add(entity1)
            #Case 2: e2 is not spurious
            if len(s2) > 0:
              qn_entities_clean.add(entity2)
            #Case 3: Both turn out to be spurious, pick the one with lower score, (less freq. stopwords)
            if len(qn_entities_clean) == 0:
              if score1 < score2:
                qn_entities_clean.add(entity1)
              else:
                qn_entities_clean.add(entity2)
      qn_entities = qn_entities_clean
    return list(qn_entities)


  def get_question_entities(self, question):
    qn_entities = []
    q_words = question.split(" ")
    max_gram = ""
    for n in range(1, len(q_words) + 1):
      i = 0
      while i + n <= len(q_words):
        gram = q_words[i:i + n]
        gram = " ".join(gram)
        if gram in self.valid_entities_set:
          qn_entities.append(gram)
        i = i + 1

    #remove stop entities, substrings, spurious entities
    qn_entities = self.remove_all_stopwords_except_one(qn_entities)
    qn_entities = self.remove_substrings(qn_entities)
    qn_entities = self.remove_spurious_entities(qn_entities, question)
    return qn_entities