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
    logits = self.build_model() #batch_size * count_entities
    self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.answer))
    self.optimizer = tf.train.AdamOptimizer().minimize(self.loss_op)
    self.predict_op = tf.argmax(logits, 1, name="predict_op")
    init_op = tf.initialize_all_variables()
    self.sess.run(init_op)


  def build_inputs(self):
    flags = tf.app.flags
    batch_size = flags.FLAGS.batch_size
    self.question = tf.placeholder(tf.int32, [None, self.size[QUESTION]], name="question")
    self.qn_entities = tf.placeholder(tf.int32, [None, self.size[QN_ENTITIES]], name="qnEntities")
    self.answer = tf.placeholder(tf.int32, shape=[None], name="answer")
    self.keys = tf.placeholder(tf.int32, [None, self.size[KEYS], 2], name="keys")
    self.values = tf.placeholder(tf.int32, [None, self.size[VALUES]], name="values")
    self.dropout_memory = tf.placeholder(tf.float32)


  def build_params(self):
    flags = tf.app.flags
    embedding_size = flags.FLAGS.embedding_size
    hops = flags.FLAGS.hops
    with tf.variable_scope(self.name):
      nil_word_slot = tf.constant(np.zeros([1, embedding_size]), dtype=tf.float32)
      #initializer = tf.random_normal_initializer(stddev=0.1)
      initializer = tf.contrib.layers.xavier_initializer()
      E = tf.Variable(initializer([self.vocab_size, embedding_size]), name='E')
      self.A = tf.concat(0, [nil_word_slot, E]) # vocab_size+1 * embedding_size
      self.B = tf.Variable(initializer([embedding_size, self.count_entities]), name='B')
      self.R_list = []
      for k in  xrange(hops):
        R_k = tf.Variable(initializer([embedding_size, embedding_size]), name='H')
        self.R_list.append(R_k)


  def build_model(self):
    flags = tf.app.flags
    hops = flags.FLAGS.hops
    batch_size = flags.FLAGS.batch_size
    memory_size = self.size[KEYS]

    with tf.variable_scope(self.name):
      #this was leading to poorer performance
      #self.reset_updates_to_nil_word_embedding()
      q_emb = tf.nn.embedding_lookup(self.A, self.question) #batch_size * size_question * embedding_size
      q_0 = tf.reduce_sum(q_emb, 1) #batch_size * embedding_size
      q = [q_0]

      for hop in xrange(hops):
        keys_emb = tf.nn.embedding_lookup(self.A,
                                          self.keys)  # batch_size * size_memory * 2 * embedding_size
        k = tf.reduce_sum(keys_emb, 2)  # batch_size * size_memory * embedding_size

        #apply dropout on keys
        ones = tf.ones([memory_size, 1], tf.float32)
        ones_dropout = tf.nn.dropout(ones, self.dropout_memory, noise_shape=[memory_size, 1])
        #k_dropout = k * ones_dropout

        q_temp = tf.expand_dims(q[-1],-1) # batch_size * embedding_size * 1
        q_temp1 = tf.transpose(q_temp, [0, 2, 1])  # batch_size * 1 * embedding_size
        prod = k * q_temp1  # batch_size * size_memory * embedding_size
        dotted = tf.reduce_sum(prod, 2) # batch_size * size_memory
        probs = tf.nn.softmax(dotted) # batch_size * size_memory

        values_emb = tf.nn.embedding_lookup(self.A, self.values) #batch_size * size_memory * embedding_size

        #apply dropout on values
        values_emb_dropout = values_emb * ones_dropout

        probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1]) #batch_size * 1 * size_memory
        v_temp = tf.transpose(values_emb_dropout, [0,2,1]) #batch_size * embedding_size * size_memory
        o_k = tf.reduce_sum(v_temp * probs_temp, 2) #batch_size * embedding_size

        R_k = self.R_list[hop]
        R_1 = self.R_list[0] #Reuse the R matrix
        q_k = tf.matmul(q[-1], R_k) + o_k
        q.append(q_k)
      return tf.matmul(q_k, self.B)


  def batch_fit(self, batch_dict):
    flags = tf.app.flags
    dropout_memory = flags.FLAGS.dropout_memory
    feed_dict = {self.question: batch_dict[QUESTION],
                 self.qn_entities: batch_dict[QN_ENTITIES],
                 self.answer: batch_dict[ANSWER],
                 self.keys: batch_dict[KEYS],
                 self.values: batch_dict[VALUES],
                 self.dropout_memory: dropout_memory}
    self.sess.run(self.optimizer, feed_dict=feed_dict)
    loss = self.sess.run(self.loss_op, feed_dict=feed_dict)
    return loss


  def predict(self, batch_dict):
    feed_dict = {self.question: batch_dict[QUESTION],
                 self.qn_entities: batch_dict[QN_ENTITIES],
                 self.answer: batch_dict[ANSWER],
                 self.keys: batch_dict[KEYS],
                 self.values: batch_dict[VALUES],
                 self.dropout_memory: 1.0}
    return self.sess.run(self.predict_op, feed_dict=feed_dict)


  def get_embedding_matrix(self):
    return self.sess.run(self.A)

  def get_nil_word_embedding(self):
    indices = [0]
    return self.sess.run(tf.gather(self.A, indices))

  # #scatter_update could only be applied to Variable types :(
  # def reset_updates_to_nil_word_embedding(self):
  #   flags = tf.app.flags
  #   embedding_size = flags.FLAGS.embedding_size
  #   nil_word_slot = tf.zeros([1, embedding_size])
  #   row1_n = tf.gather(self.A, range(1, self.vocab_size))
  #   self.A = tf.concat(0, [nil_word_slot, row1_n])




