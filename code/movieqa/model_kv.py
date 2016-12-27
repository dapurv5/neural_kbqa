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
ANSWER = "answer"
KEYS = "keys"
VALUES = "values"

class KeyValueMemNN(object):
  def __init__(self, sess, size, idx_size, entity_idx_size):
    self.sess = sess
    self.size = size
    self.name = "KeyValueMemNN"
    self.vocab_size = idx_size
    self.count_entities = entity_idx_size
    self.build_inputs()
    self.build_params()
    logits = self.build_model()
    self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.answer))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss_op)
    self.predict_op = tf.argmax(logits, 1, name="predict_op")
    init_op = tf.initialize_all_variables()
    self.sess.run(init_op)


  def build_inputs(self):
    flags = tf.app.flags
    batch_size = flags.FLAGS.batch_size
    self.question = tf.placeholder(tf.int32, [batch_size, self.size[QUESTION]], name="question")
    self.qn_entities = tf.placeholder(tf.int32, [batch_size, self.size[QN_ENTITIES]], name="qnEntities")
    self.answer = tf.placeholder(tf.int32, shape=[batch_size], name="answer")
    self.keys = tf.placeholder(tf.int32, [batch_size, self.size[KEYS], 2], name="keys")
    self.values = tf.placeholder(tf.int32, [batch_size, self.size[VALUES]], name="values")


  def build_params(self):
    flags = tf.app.flags
    embedding_size = flags.FLAGS.embedding_size
    with tf.variable_scope(self.name):
      #nil_word_slot = tf.zeros([1, embedding_size], tf.int32)
      initializer = tf.random_normal_initializer(stddev=0.1)
      self.A = tf.Variable(initializer([self.vocab_size+1, embedding_size]), name='A') #embedding matrix
      #A = tf.concat(0, [nil_word_slot, A_]) # vocab_size+1 * embedding_size
      #self.A = tf.Variable(A, name="A")
      self.B = tf.Variable(initializer([embedding_size, self.count_entities]), name='B')
      self.H = tf.Variable(initializer([embedding_size, embedding_size]), name='H')


  def build_model(self):
    with tf.variable_scope(self.name):
      q_emb = tf.nn.embedding_lookup(self.A, self.question) #batch_size * size_question * embedding_size
      q_0 = tf.reduce_sum(q_emb, 1) #batch_size * embedding_size
      q = [q_0]

      for hop in xrange(2):
        keys_emb = tf.nn.embedding_lookup(self.A,
                                          self.keys)  # batch_size * size_memory * 2 * embedding_size
        k = tf.reduce_sum(keys_emb, 2)  # batch_size * size_memory * embedding_size

        q_temp = tf.expand_dims(q[-1],-1) # batch_size * embedding_size * 1
        q_temp1 = tf.transpose(q_temp, [0, 2, 1])  # batch_size * 1 * embedding_size
        prod = k * q_temp1  # batch_size * size_memory * embedding_size
        dotted = tf.reduce_sum(prod, 2) # batch_size * size_memory
        probs = tf.nn.softmax(dotted) # batch_size * size_memory

        values_emb = tf.nn.embedding_lookup(self.A, self.values) #batch_size * size_memory * embedding_size
        probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1]) #batch_size * 1 * size_memory
        v_temp = tf.transpose(values_emb, [0,2,1]) #batch_size * embedding_size * size_memory
        o_k = tf.reduce_sum(v_temp * probs_temp, 2) #batch_size * embedding_size

        q_k = tf.matmul(q[-1], self.H) + o_k
        q.append(q_k)
      return tf.matmul(q_k, self.B)


  def batch_fit(self, batch_dict):
    feed_dict = {self.question: batch_dict[QUESTION],
                 self.qn_entities: batch_dict[QN_ENTITIES],
                 self.answer: batch_dict[ANSWER],
                 self.keys: batch_dict[KEYS],
                 self.values: batch_dict[VALUES]}
    self.sess.run(self.optimizer, feed_dict=feed_dict)
    loss = self.sess.run(self.loss_op, feed_dict=feed_dict)
    return loss


  def predict(self, batch_dict):
    feed_dict = {self.question: batch_dict[QUESTION],
                 self.qn_entities: batch_dict[QN_ENTITIES],
                 self.answer: batch_dict[ANSWER],
                 self.keys: batch_dict[KEYS],
                 self.values: batch_dict[VALUES]}
    return self.sess.run(self.predict_op, feed_dict=feed_dict)


