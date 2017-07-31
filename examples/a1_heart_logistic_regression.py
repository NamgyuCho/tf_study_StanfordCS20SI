import os
import tensorflow as tf
import time
import sys
import heart_dataset

# Define parameters for the model
learning_rate = 1e-3
batch_size = 36
n_epochs = 500

# Step 1: Read in heart data
print('1. Read HEART data')
# Feature dimension is nine and the total number of samples is 462
heart_db = heart_dataset.read_data_sets('./data/heart.txt')
n_in_dim = 8
n_out_dim = 2


# Step 2: Create placeholder for features and labels
print('2. Create placeholders')
X = tf.placeholder(tf.float32, shape=[batch_size, n_in_dim], name='X_placeholder')
Y = tf.placeholder(tf.int32, shape=[batch_size, n_out_dim], name='Y_placeholder')


# Step 3: Create weights and bias
w = tf.Variable(tf.random_normal(shape=[n_in_dim, n_out_dim], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([n_out_dim]), name='bias')


# Step 4: Build a model
logits = tf.matmul(X, w) + b


# Step 5: Define a loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='loss')
loss = tf.reduce_mean(entropy)


# Step 6: Define a training op
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# Step 7: Start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/03/heart_logistic_reg', sess.graph)

    # Step 8: Train the model
    for i in range(n_epochs):
        total_loss = 0

        num_batches = heart_db.train.num_batches(batch_size)
        for _ in range(num_batches):
            X_batch, Y_batch = heart_db.train.next_batch(batch_size)
            tmp, l = sess.run([train_op, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, float(total_loss)/num_batches))
    print('Optimization finished')

    # Step 9: Test the model
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    total_correct_preds = 0

    num_batches = heart_db.test.num_batches(batch_size)
    for i in range(num_batches):
        X_batch, Y_batch = heart_db.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds/num_batches))

    writer.close()


