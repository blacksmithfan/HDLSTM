
'''
Re-implementation of Heat Diffusion Long Short-Term Memory with TensorFlow
'''

from __future__ import print_function
import scipy.io as spio
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from dataset import Dataset
import numpy as np


# Parameters
learning_rate = 0.001
training_iters = 20000000
batch_size = 512
display_step = 10

# Network Parameters
n_input = 128 # histogram dimension
n_steps = 101 # timesteps
n_hidden = 30 # hidden layer num of features
n_classes = 55 # 3D shape classes



# define tf placeholders
x = tf.placeholder("float", [n_steps, n_input, None])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def HD_LSTM(x, weights, biases):


    x = tf.transpose(x, [0, 2, 1])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = HD_LSTM(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

dataset = Dataset(n_classes=n_classes, train_path='ShapeNet_hist_train/', test_path='ShapeNet_hist_test/', shuffleType='normal', seqLength=0, CNN_type='Alex')

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:

        batch_data, label_data = dataset.next_batch(batch_size, 'train')

        batch_x = batch_data
        batch_y = label_data

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # # Calculate accuracy for the query data (Only use a simple recognition task
    # for evaluating the learned shape representation)
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: batch_data, y: label_data}))
