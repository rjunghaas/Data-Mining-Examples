'''
Modified basic mini-batch SGD classifier with following configuration to improve performance:
Initialization = He Initialization
Activation Function = ELU
Normalization = Batch Normalization
Regularization = Dropout
Optimizer = Adam
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout

# Load MNIST data
mnist = input_data.read_data_sets("/tmp/data")

# Constants for setting up neural network
n_inputs = 28*28 #MNIST
n_hidden1 = 30
n_hidden2 = 10
n_outputs = 10

# Learning rate for optimizer
learning_rate = 0.01

# Parameters for training
n_epochs = 10
batch_size = 50

# Parameter for Dropout
keep_prob = 0.5

# Create placeholders to hold X & y values in neural network
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Set up for Batch Normalization
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

# Construction phase of neural network.  Feed X into hidden1 into hidden2 and then output
with tf.name_scope("dnn"):
    # Use He Initialization
    he_init = tf.contrib.layers.variance_scaling_initializer()
    # Add hidden layers for dropout
    X_drop = dropout(X, keep_prob, is_training=is_training)
    # Change activation functions to ELU and apply batch normalization
    hidden1 = fully_connected(X_drop, n_hidden1, scope="hidden1", weights_initializer=he_init, activation_fn=tf.nn.elu, normalizer_fn=batch_norm, normalizer_params=bn_params)
    hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)
    hidden2 = fully_connected(hidden1_drop, n_hidden2, scope="hidden2", weights_initializer=he_init, activation_fn=tf.nn.elu, normalizer_fn=batch_norm, normalizer_params=bn_params)
    hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
    logits = fully_connected(hidden2_drop, n_outputs, scope="outputs", activation_fn=None, normalizer_fn=batch_norm, normalizer_params=bn_params)

# Use cross entropy for loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Use optimizer to minimize cross entropy loss function
with tf.name_scope("train"):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Use Adam Optimizer instead of Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

# Accuracy used as evaluation measure to pass to optimizer
with tf.name_scope("eval"):
    # Get highest logit as class our neural network predicts and form a tensor with boolean of whether prediction was correct
    correct = tf.nn.in_top_k(logits, y, 1)
    # Cast from boolean to float and compute average of correct tensor to get overall accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize variables and train neural network
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()

    for epoch in range(n_epochs):
        # Grab mini-batch and pass to train network.  Have to specify in feed_dict is_training is True for batch normalization
        for iteration in range(mnist.train.num_examples):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})

        # Test model against batch of test data for accuracy.  Have to specify in feed_dict is_training is False for batch normalization
        acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", epoch, "Test accuracy:", acc_test)
