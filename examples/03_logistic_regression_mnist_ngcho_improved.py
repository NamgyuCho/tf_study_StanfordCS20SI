""" Starter code for logistic regression model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import sys

# Define paramaters for the model
learning_rate = 0.01
batch_size =20
n_epochs = 30

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
print('Read MNIST data')
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# Features are of the type float, and labels are of the type int
print('Create placeholders')
X = tf.placeholder(tf.float32, shape=[batch_size, 28*28], name='X_placeholder')
Y = tf.placeholder(tf.int32, shape=[batch_size, 10], name='Y_placeholder')

# Step 4: build model
# the model that returns the logits.
print('Build the conv network')
X_image = tf.reshape(X, [batch_size, 28, 28, 1]) # Originally, X was of shape [batchsize, w, h]

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(X_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.01))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.01))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv, name='loss')
loss = tf.reduce_mean(entropy)

# Step 6: define training op
# using gradient descent to minimize loss
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


with tf.Session() as sess:
    print('Start session')
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/03/mnist_conv', sess.graph)
    n_batches = int(mnist.train.num_examples/batch_size)

    # step 8: train the model
    for i in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            batch = mnist.train.next_batch(batch_size)
            _, l = sess.run([train_op, loss], feed_dict={X: batch[0], Y: batch[1], keep_prob:0.5})
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss/n_batches))

    print('Optimization finished')

    # test the model
    correct_preds = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(

    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch, keep_prob: 1.0})
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

    writer.close()
