#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time

# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# correct labels
y_ = tf.placeholder(tf.float32, [None, 10])

# define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define a training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# train
print("Starting the training...")
start_time = time()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("The training took %.4f seconds." % (time() - start_time))

# validate
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: %.4f" % (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
