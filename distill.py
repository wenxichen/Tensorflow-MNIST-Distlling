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

def main():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model for large net
  # initialize parameters
  x = tf.placeholder(tf.float32, [None, 784])
  keep_prob = tf.placeholder(tf.float32)
  L_W1 = tf.Variable(tf.truncated_normal([784, 1200], stddev=0.1), name='L_w1')
  L_W2 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.1), name='L_w2')
  L_W3 = tf.Variable(tf.truncated_normal([1200, 10], stddev=0.1), name='L_w3')
  L_b1 = tf.Variable([0.1]*1200, name='L_b1')
  L_b2 = tf.Variable([0.1]*1200, name='L_b2')
  L_b3 = tf.Variable([0.1]*10, name='L_b3')
  # temperature
  T = tf.constant(5.0)

  # first hidden layer
  L_a1 = tf.nn.relu(tf.matmul(x, L_W1) + L_b1)
  L_a1 = tf.nn.dropout(L_a1, keep_prob)

  # second hidden layer
  L_a2 = tf.nn.relu(tf.matmul(L_a1, L_W2) + L_b2)
  L_a2 = tf.nn.dropout(L_a2, keep_prob)

  # output layer
  L_y = tf.matmul(L_a2, L_W3) + L_b3

  L_q = tf.nn.softmax(L_y * (1.0/T))

  # Define loss and optimizer

  # Create the model for small net
  # initialize parameters
  y_ = tf.placeholder(tf.float32, [None, 10])
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

  q = tf.nn.softmax(y * (1.0/T))

  loss = tf.reduce_mean( - tf.reduce_sum(tf.log(q) * tf.stop_gradient(L_q), -1), -1)

  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

  # init_op = tf.variables_initializer([W1, W2, W3, b1, b2, b3])

  saver = tf.train.Saver({'L_w1': L_W1, 'L_w2': L_W2, 'L_w3': L_W3, 'L_b1': L_b1, 'L_b2': L_b2, 'L_b3': L_b3})
  saver2 = tf.train.Saver({'dis_w1': W1, 'dis_w2': W2, 'dis_w3': W3, 'dis_b1': b1, 'dis_b2': b2, 'dis_b3': b3})

  with tf.Session() as sess:
  	# sess.run(init_op)
    saver.restore(sess, "path_to_load_large_net")
    saver2.restore(sess, "path_to_load_small_net")
    print("Model restored.")

    # Train
    for i in range(1,10001):
      batch_xs, batch_ys = mnist.train.next_batch(500)
      _, train_loss = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

      if i % 10 == 0:
        print("train loss:", train_loss)

    save_path = saver2.save(sess, "path_to_save_distlled_ckpt")
    print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  main()










