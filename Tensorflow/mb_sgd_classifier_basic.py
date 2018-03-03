import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

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

# Create placeholders to hold X & y values in neural network
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Construction phase of neural network.  Feed X into hidden1 into hidden2 and then output
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

# Use cross entropy for loss function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Use Gradient Descent to minimize cross entropy loss function
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Accuracy used as evaluation measure to pass to optimizer
with tf.name_scope("eval"):
    # Get highest logit as class our neural network predicts and form a tensor with boolean of whether prediction was correct
    correct = tf.nn.in_top_k(logits, y, 1)
    # Cast from boolean to float and compute average of correct tensor
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize variables and train neural network
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()

    for epoch in range(n_epochs):
        # Grab mini-batch and pass to training network
        for iteration in range(mnist.train.num_examples):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        # Test model against batch of test data for accuracy
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", epoch, "Test accuracy:", acc_test)
