import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST Data
mnist = input_data.read_data_sets("/tmp/data")

# Constants for setting up neural network
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

# Learning rate for optimizer
learning_rate = 0.01

# Parameters for training
n_epochs = 10
batch_size = 50

# Placeholders for X & y values.  First value of X placeholder is for batch size
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# Construction phase of neural network
with tf.name_scope("rnn"):
    # Create cells using BasicLSTMCell factory
    lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]

    # Stack multiple LSTM cells in a single layer to make deep RNN
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    # Pass cells to dynamic_rnn to construct neural network.  Returns output tensors for each time step and final state of network
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

    # Take final state of network
    top_layer_h_state = states[-1][1]

    # Add Softmax layer and pass it final state to compute output
    logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")

# Use cross entropy for loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Use Adam Optimizer to minimize cross entropy loss function
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

# Accuracy used as evaluation measure to pass to optimizer
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize variables and train neural network
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        # Test model against test data for accuracy
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images.reshape((-1, n_steps, n_inputs)), y: mnist.test.labels})
        print ("Epoch", epoch, "Test Accuracy", acc_test)
