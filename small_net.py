# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from mnist import input_data

import tensorflow as tf

FLAGS = None
LR = 0.05

def main():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  # initialize parameters
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  keep_prob = tf.placeholder(tf.float32)
  W1 = tf.Variable(tf.truncated_normal([784, 20], stddev=0.1), name='dis_w1')
  W2 = tf.Variable(tf.truncated_normal([20, 20], stddev=0.1), name='dis_w2')
  W3 = tf.Variable(tf.truncated_normal([20, 10], stddev=0.1), name='dis_w3')
  b1 = tf.Variable([0.1]*20, name='dis_b1')
  b2 = tf.Variable([0.1]*20, name='dis_b2')
  b3 = tf.Variable([0.1]*10, name='dis_b3')

  # first hidden layer
  a1 = tf.nn.relu(tf.matmul(x, W1) + b1)
  a1 = tf.nn.dropout(a1, keep_prob)

  # second hidden layer
  a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
  a2 = tf.nn.dropout(a2, keep_prob)

  # output layer
  y = tf.matmul(a2, W3) + b3

  # Define loss and optimizer

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

  # initialize variables or load from ckpt
  # sess = tf.InteractiveSession()
  # init_op = tf.global_variables_initializer()
  saver2 = tf.train.Saver({'dis_w1': W1, 'dis_w2': W2, 'dis_w3': W3, 'dis_b1': b1, 'dis_b2': b2, 'dis_b3': b3})

  # Train
  with tf.Session() as sess:
    # sess.run(init_op)
    saver2.restore(sess, "path_to_load_ckpt")
    for i in range(1,3001):
      batch_xs, batch_ys = mnist.train.next_batch(55000)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
      if i % 10 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_acc = sess.run(accuracy, feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        val_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y_: mnist.test.labels,
                                                keep_prob: 1.0})

        print('(iter {}) train acc: {:.4f}, val acc: {:.4f}'.format(i, train_acc, val_acc))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  main()
