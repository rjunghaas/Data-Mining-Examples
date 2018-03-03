# Inspired by:
# https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-4-logistic-regression-2afd0cabc54

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math

# Load MNIST data
mnist = input_data.read_data_sets("/tmp/data")

# Constants for setting up batches
n_steps = 28
n_inputs = 28*28
n_classes = 10
n_epochs = 3
batch_size = 50
learning_rate = 0.0001

def createOneHot(batch):
    one_hot = []
    row = np.zeros([n_classes])
    for each in batch:
        row[each] = 1
        one_hot.append(row)
        row = np.zeros([n_classes])

    return one_hot

# Create placeholders for x and y values
X = tf.placeholder(tf.float32, [None, n_inputs], name="X")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")

# Create variables for model weights
W = tf.Variable(tf.zeros([n_inputs, n_classes]), name="Weights")
b = tf.Variable(tf.zeros([n_classes]), name="bias")

# Make prediction and convert to probability distribution
pred = tf.nn.softmax(tf.matmul(X, W) + b)
pred_prob = tf.divide(pred, tf.exp(tf.reduce_logsumexp(pred)))  # Calculate sum of log-values and use as our denominator

# Cost function
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred_prob), reduction_indices=1))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(cost)

# Use accuracy to evaluate model
correct = tf.equal(tf.argmax(pred_prob, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize variables and train neural network
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(n_epochs):
        # Get mini-batch pass to train network
        for batch_index in range(mnist.train.num_examples):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Transform y_batch to one-hot vector
            y_batch_one_hot = createOneHot(y_batch)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch_one_hot})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: createOneHot(mnist.test.labels)})
        print("Epoch:", epoch, "accuracy:", acc_test)

    # Display results
    print("W:")
    print(sess.run(W))
    print("b:")
    print(sess.run(b))
