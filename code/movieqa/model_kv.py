#!/usr/bin/python

import argparse
import os
import numpy as np
import tensorflow as tf

QUESTION = "question"
QN_ENTITIES = "qn_entities"
ANS_ENTITIES = "ans_entities"
SOURCES = "sources"
RELATIONS = "relations"
TARGETS = "targets"

class KeyValueMemNN(object):
  def __init__(self, size):
    self.size = size

