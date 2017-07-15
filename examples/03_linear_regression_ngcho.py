import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# Both have the type float32
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
# name your variables w and b
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='biases')


# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
# name your variable loss
# Assign: define Huber loss
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    res = tf.cond(residual < delta, lambda: 0.5 * tf.square(residual), lambda: delta * residual - 0.5 * tf.square(delta))
    return res

# original
#loss = (Y - Y_predicted) ** 2
loss = huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)



# Phase 2: Train our model
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    # TO - DO
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/03/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(50): # run 100 epochs
        total_loss = 0
        for x, y in data:
            # Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
            # TO DO: write sess.run()
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})

            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss/n_samples))
    # plot the results
    X, Y = data.T[0], data.T[1]
    plt.plot(X, Y, 'bo', label='Real data')
    w, b = sess.run([w, b])
    plt.plot(X, X * w + b, 'r', label='Predicted data')
    plt.legend()
    plt.show()

    writer.close()


