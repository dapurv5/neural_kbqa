#!/usr/bin/python

import argparse
import os
import numpy as np
import tensorflow as tf

from dataset_reader import DatasetReader


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
flags.DEFINE_integer("epochs", 3, "Number of epochs to train for.")
flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")


FLAGS = flags.FLAGS

graph = tf.Graph()
with graph.as_default():
  pass

def main(args):
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  args.input_examples = args.train_examples
  train_examples = DatasetReader(args).get_examples()
  args.input_examples = args.test_examples
  test_examples = DatasetReader(args).get_examples()

  batch_size = FLAGS.batch_size
  num_train = len(train_examples)
  batches = zip(range(0, num_train - batch_size, batch_size), range(batch_size, num_train, batch_size))
  batches = [(start, end) for start, end in batches]
  batches = batches[0:1]
  print batches

  with tf.Session(graph=graph) as sess:
    model = None
    for epoch in range(1, FLAGS.epochs+1):
      np.random.shuffle(batches)
      total_cost = 0.0
      for start, end in batches:
        print start, end

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--train_examples', help='the train file', required=True)
  parser.add_argument('--test_examples', help='the test file', required=True)
  parser.add_argument('--dev_examples', help='the dev file', required=True)
  parser.add_argument('--word_idx', help='word vocabulary', required=True)
  parser.add_argument('--entity_idx', help='entity vocabulary', required=True)
  parser.add_argument('--relation_idx', help='relation vocabulary', required=True)
  args = parser.parse_args()
  main(args)
