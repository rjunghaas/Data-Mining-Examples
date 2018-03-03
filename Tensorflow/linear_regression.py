import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Constants for setting up batches
n_epochs = 10
batch_size = 100
learning_rate = 0.0001
training_pct = 0.7

# Helper function that will randomly select items and construct batches
def fetch_batch(epoch, batch_index, batch_size):
  np.random.seed(epoch * n_batches + batch_index)
  indices = np.random.randint(m, size=batch_size)
  X_batch = x_scaled_data[indices]
  y_batch = y_scaled_data.reshape(-1, 1)[indices]
  return X_batch, y_batch

# Read in CSV file
housing = pd.DataFrame.from_csv('housing.csv')

# Create one hot encoding for ocean_proximity values
one_hot = pd.get_dummies(housing['ocean_proximity'])
housing = housing.drop('ocean_proximity', axis=1)
housing = housing.join(one_hot)
housing['total_bedrooms'].fillna(0, inplace=True)

# Create variables for x and y values
y_data = housing['median_house_value']
x_data = housing.drop('median_house_value', axis=1)

# Calculate dimensions of data and use for batch size
n = len(x_data.columns)
m = len(x_data)
n_batches = int(np.ceil(m / batch_size))

# Normalize data
scaler = StandardScaler()
x_scaled_data = scaler.fit_transform(x_data)
y_scaled_data = scaler.fit_transform(y_data)

# Create placeholders for X & y values in neural network
X = tf.placeholder(tf.float32, shape=(batch_size, n), name="X")
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")

# Initialize W and b
W = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0, seed=42), name="Weights")
b = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0, seed=30), name="bias")

# Calculate Mean Square Error
with tf.name_scope("loss"):
    y_pred = tf.add(tf.matmul(X, W), b)
    mse = tf.reduce_mean(tf.square(y - y_pred), name="mse")

# Use optimizer to minimize mse
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

# Initialize variables and train neural network
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(n_epochs):
        # Get mini-batch pass to train network
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        mse_test = mse.eval(feed_dict={X: X_batch, y: y_batch})
        print("Epoch:", epoch, "mse:", mse_test)

    # Display results
    print("W:")
    print(sess.run(W))
    print("b:")
    print(sess.run(b))
