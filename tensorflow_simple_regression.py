import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import time
import numpy as np

# Get the data
total_features, total_prices = load_boston(True)

# Keep 300 samples for training
train_features = scale(total_features[:300])
train_prices = total_prices[:300]

# Keep 100 samples for validation
valid_features = scale(total_features[300:400])
valid_prices = total_prices[300:400]

# Keep remaining samples as test set
test_features = scale(total_features[400:])
test_prices = total_prices[400:]

train_features =  np.float32(train_features)#tf.cast(train_features, tf.float32)
valid_features = np.float32(valid_features)
valid_features = np.float32(test_features)
train_prices = np.float32(total_prices[:300])
valid_prices = np.float32(total_prices[300:400])
test_prices = np.float32(total_prices[400:])

w = tf.Variable(tf.truncated_normal([13, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.zeros(1, dtype = tf.float32))

def calc(x, y):
    # Returns predictions and error
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions))
    return [ predictions, error ]

y, cost = calc(train_features, train_prices)
# Feel free to tweak these 2 values:
learning_rate = 0.025
epochs = 30000
points = [[], []] # You'll see later why I need this

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:

    sess.run(init)
    startTime = time.time()
    for i in list(range(epochs)):

        sess.run(optimizer)

        if i % 10 == 0.:
            points[0].append(i+1)
            points[1].append(sess.run(cost))

        if i % 1000 == 0:
            print(sess.run(cost))
    endTime = time.time()
    print('Training time: ', endTime - startTime)
##    plt.plot(points[0], points[1], 'r--')
##    plt.axis([0, epochs, 50, 600])
##    plt.show()
##
##    valid_cost = calc(valid_features, valid_prices)[1]
##
##    print('Validation error =', sess.run(valid_cost), '\n')
##
##    test_cost = calc(test_features, test_prices)[1]
##
##    # print('Test error =', sess.run(test_cost), '\n')
