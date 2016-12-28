#!/usr/bin/python

import argparse
import os
import numpy as np
import random
import tensorflow as tf
import tqdm

from kv_dataset_reader import DatasetReader
from kv_dataset_reader import get_maxlen
from model_kv import KeyValueMemNN

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
flags.DEFINE_integer("save_interval", 10, "save model every x epochs")
flags.DEFINE_integer("batch_size", 256, "Batch size for training.")
flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
flags.DEFINE_integer("epochs", 1000, "Number of epochs to train for.")
flags.DEFINE_integer("embedding_size", 200, "Embedding size for embedding matrices.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")


FLAGS = flags.FLAGS
QUESTION = "question"
QN_ENTITIES = "qn_entities"
ANS_ENTITIES = "ans_entities"
SOURCES = "sources"
RELATIONS = "relations"
TARGETS = "targets"
ANSWER = "answer"
KEYS = "keys"
VALUES = "values"


def prepare_batch(batch_examples):
  batch_size = FLAGS.batch_size
  batch_dict = {}
  batch_dict[QUESTION] = gather_single_column_from_batch(batch_examples, QUESTION)
  batch_dict[QN_ENTITIES] = gather_single_column_from_batch(batch_examples, QN_ENTITIES)
  batch_dict[SOURCES] = gather_single_column_from_batch(batch_examples, SOURCES)
  batch_dict[RELATIONS] = gather_single_column_from_batch(batch_examples, RELATIONS)
  batch_dict[TARGETS] = gather_single_column_from_batch(batch_examples, TARGETS)
  batch_dict[KEYS] = gather_key_from_batch(batch_examples)
  batch_dict[VALUES] = gather_single_column_from_batch(batch_examples, TARGETS)
  labels = np.zeros([batch_size])
  for i in xrange(batch_size):
    labels[i] = random.sample(batch_examples[i][ANS_ENTITIES], 1)[0]
  batch_dict[ANSWER] = labels
  return batch_dict


def gather_single_column_from_batch(batch_examples, column_name):
  batch_size = FLAGS.batch_size
  column = []
  for i in xrange(batch_size):
    column.append(batch_examples[i][column_name])
  return np.array(column) #batch_size * maxlen(column_name)

def gather_key_from_batch(batch_examples):
  batch_size = FLAGS.batch_size
  column = []
  for i in xrange(batch_size):
    assert(len(batch_examples[i][SOURCES] == len(batch_examples[i][RELATIONS])))
    memory_length = len(batch_examples[i][SOURCES])
    memories = []
    src = batch_examples[i][SOURCES]
    rel = batch_examples[i][RELATIONS]
    for memory_index in xrange(memory_length):
      memories.append(np.array([src[memory_index], rel[memory_index]]))
    column.append(np.array(memories))
  return np.array(column) #batch_size * memory_length * 2


def save_model(sess):
  saver = tf.train.Saver()
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "model_kv.ckpt"))
  print("Model saved in file: %s" % save_path)


def main(args):
  maxlen = get_maxlen(args.train_examples, args.test_examples, args.dev_examples)
  maxlen[KEYS], maxlen[VALUES] = maxlen[SOURCES], maxlen[SOURCES]

  args.input_examples = args.train_examples
  train_reader = DatasetReader(args, maxlen, share_idx=True)
  train_examples = train_reader.get_examples()
  args.input_examples = args.test_examples
  test_reader = DatasetReader(args, maxlen, share_idx=True)
  test_examples = test_reader.get_examples()
  args.input_examples = args.dev_examples
  dev_reader = DatasetReader(args, maxlen, share_idx=True)
  dev_examples = dev_reader.get_examples()

  batch_size = FLAGS.batch_size
  num_train = len(train_examples)
  batches = zip(range(0, num_train - batch_size, batch_size), range(batch_size, num_train, batch_size))
  batches = [(start, end) for start, end in batches]
  batches = batches[0:10] #Uncomment this to run locally
  with tf.Session() as sess:
    model = KeyValueMemNN(sess, maxlen, train_reader.get_idx_size(), train_reader.get_entity_idx_size())
    if os.path.exists(os.path.join(FLAGS.checkpoint_dir, "model_kv.ckpt")):
      saver = tf.train.Saver()
      save_path = os.path.join(FLAGS.checkpoint_dir, "model_kv.ckpt")
      saver.restore(sess, save_path)
      print("Model restored from file: %s" % save_path)

    for epoch in range(1, FLAGS.epochs+1):
      np.random.shuffle(batches)
      sum_prob_of_error = 0
      for start, end in batches:
        batch_examples = train_examples[start:end]
        batch_dict = prepare_batch(batch_examples)
        prob_of_error = model.batch_fit(batch_dict)
        predictions = model.predict(batch_dict)
        labels = tf.constant(batch_dict[ANSWER], tf.int64)
        train_accuracy = tf.contrib.metrics.accuracy(predictions, labels)
        print "EPOCH={epoch}:TRAIN_LOSS={class_loss}:TRAIN_ACC:{train_acc}".\
          format(epoch=epoch, class_loss=prob_of_error, train_acc=sess.run(train_accuracy))
        sum_prob_of_error += prob_of_error
      if epoch > 0 and epoch % FLAGS.save_interval == 0:
        save_model(sess)
      if epoch > 0 and epoch % FLAGS.evaluation_interval == 0:
        test_accuracy = get_accuracy(model, train_examples)
        print "EPOCH={epoch}:TEST_ACCURACY={test_accuracy}".format(epoch=epoch,
                                                                   test_accuracy=test_accuracy)

def get_accuracy(model , examples):
  batch_size = FLAGS.batch_size
  num_examples = len(examples)
  batches = zip(range(0, num_examples - batch_size, batch_size), range(batch_size, num_examples, batch_size))
  batches = [(start, end) for start, end in batches]
  batches = batches[0:10] #Uncomment this to run locally
  count_correct = 0.0
  count_total = 0.0
  for start, end in batches:
    batch_examples = examples[start:end]
    batch_dict = prepare_batch(batch_examples)
    predictions = model.predict(batch_dict)
    for i in xrange(len(batch_examples)):
      correct_answers = set(batch_examples[i][ANS_ENTITIES])
      if predictions[i] in correct_answers:
        count_correct = count_correct + 1.0
      count_total = count_total + 1.0
  return count_correct/count_total


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--train_examples', help='the train file', required=True)
  parser.add_argument('--test_examples', help='the test file', required=True)
  parser.add_argument('--dev_examples', help='the dev file', required=True)
  parser.add_argument('--word_idx', help='word vocabulary', required=True)
  parser.add_argument('--entity_idx', help='entity vocabulary', required=True)
  parser.add_argument('--relation_idx', help='relation vocabulary', required=True)
  parser.add_argument('--idx', help='overall vocabulary', required=True)
  args = parser.parse_args()
  main(args)
