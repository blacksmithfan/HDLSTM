import tensorflow as tf
import numpy as np
import sys
from network import *
from tensorflow.python.ops import rnn, rnn_cell
import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
class Model:
    @staticmethod
    def alexnet(_X, _dropout, n_class):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        with tf.device('/gpu:0'):
            conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
            # Layer 2 (conv-relu-pool-lrn)
        # with tf.device('/gpu:1'):
            conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
            conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
            # Layer 3 (conv-relu)
        # with tf.device('/gpu:2'):
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
            # Layer 4 (conv-relu)
            conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
            # Layer 5 (conv-relu-pool)
        # with tf.device('/gpu:3'):
            conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
            # Layer 6 (fc-relu-drop)
            fc6 = tf.reshape(pool5, [-1, 6*6*256])
            fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
            fc6 = dropout(fc6, _dropout)
            # Layer 7 (fc-relu-drop)
            fc7 = fc(fc6, 4096, 4096, name='fc7')
            fc7 = dropout(fc7, _dropout)
            # Layer 8 (fc-prob)
            fc8 = fc(fc7, 4096, n_class, relu=False, name='fc8')
        return fc8

    @staticmethod    
    def cnn_lstm(_X, _dropout, n_steps, n_hidden, weights, biases):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        with tf.device('/gpu:0'):
            conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # Layer 2 (conv-relu-pool-lrn)
        with tf.device('/gpu:1'):
            conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
            conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        # Layer 3 (conv-relu)
        with tf.device('/gpu:2'):
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
            # Layer 4 (conv-relu)
            conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
            # Layer 5 (conv-relu-pool)
            conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, _dropout)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, _dropout)
        # Layer 8 (fc-prob)
#        fc8 = fc(fc7, 4096, 40, relu=False, name='fc8')
        x = tf.reshape(fc7, [-1, n_steps, 4096])
        x = tf.transpose(x, [1, 0, 2])
        print(x.get_shape())
        x = tf.reshape(x, [-1, 4096])
        print(x.get_shape())
        x = tf.split(0, n_steps, x)

        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        #Attempt to add dropout layer
        # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)

        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights) + biases

    @staticmethod    
    def cnn_lstm_multi_layer(_X, _dropout, n_steps, n_hidden, weights, biases, batch_size):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        with tf.device('/gpu:0'):
            conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # Layer 2 (conv-relu-pool-lrn)
        with tf.device('/gpu:1'):
            conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
            conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        # Layer 3 (conv-relu)
        with tf.device('/gpu:2'):
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
            # Layer 4 (conv-relu)
            conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
            # Layer 5 (conv-relu-pool)
            conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, _dropout)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, _dropout)
        # Layer 8 (fc-prob)
#        fc8 = fc(fc7, 4096, 40, relu=False, name='fc8')
        x = tf.reshape(fc7, [-1, n_steps, 4096])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, 4096])
        print(x.get_shape())
        x = tf.split(0, n_steps, x)
        with tf.variable_scope('lstm1'):
            lstm_cell_1 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            outputs_1, states_1 = rnn.rnn(lstm_cell_1, x, dtype=tf.float32)
        #Attempt to add dropout layer
        # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        # stacked_output = tf.zeros([n_steps, batch_size, n_hidden])
        # for i in range(n_steps):
        #     stacked_output[i,:,:] = outputs_1[i]
        # outputs_1 = tf.reshape(outputs_1,[n_steps, batch_size, n_hidden])
        # outputs_1 = tf.reshape(outputs_1, [-1, n_hidden])
        # outputs_1 = tf.split(0, n_steps, outputs_1)

        with tf.variable_scope('lstm2'):
            lstm_cell_2 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            outputs_2, states_2 = rnn.rnn(lstm_cell_2, outputs_1, dtype=tf.float32)

        return tf.matmul(tf.concat(1,[outputs_1[-1], outputs_2[-1]]), weights) + biases


        # return outputs[-1]

    @staticmethod    
    def cnn_lstm_dual(_X, _dropout, n_steps, n_hidden, weights, biases):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # Layer 2 (conv-relu-pool-lrn)
        conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        # Layer 3 (conv-relu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # Layer 4 (conv-relu)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        # Layer 5 (conv-relu-pool)
        conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, _dropout)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, _dropout)
        # Layer 8 (fc-prob)
#        fc8 = fc(fc7, 4096, 40, relu=False, name='fc8')
        x = tf.reshape(fc7, [-1, n_steps, 4096])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, 4096])
        x = tf.split(0, n_steps, x)

        x1 = tf.reshape(fc6, [-1, n_steps, 4096])
        x1 = tf.transpose(x1, [1, 0, 2])
        x1 = tf.reshape(x1, [-1, 4096])
        x1 = tf.split(0, n_steps, x1)
        with tf.variable_scope('lstm1'):
            lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        with tf.variable_scope('lstm2'):
            lstm_cell1 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            outputs1, states1 = rnn.rnn(lstm_cell1, x1, dtype=tf.float32)    
        #Attempt to add dropout layer
        # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        # lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        return tf.matmul(tf.concat(1,[outputs[-1], outputs1[-1]]), weights) + biases


    @staticmethod    
    def cnn_lstm_biDirection(_X, _dropout, n_steps, n_hidden, weights, biases):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # Layer 2 (conv-relu-pool-lrn)
        conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        # Layer 3 (conv-relu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # Layer 4 (conv-relu)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        # Layer 5 (conv-relu-pool)
        conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, _dropout)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, _dropout)
        # Layer 8 (fc-prob)
#        fc8 = fc(fc7, 4096, 40, relu=False, name='fc8')
        x = tf.reshape(fc7, [-1, n_steps, 4096])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, 4096])
        net = tf.split(0, n_steps, x)
        
        net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), scope='biDirection')
        net = dropout(net, 0.5)
        net = fully_connected(net, 2, activation='softmax')     
        print(net.get_shape())

        return net

    @staticmethod
    def vgg16(input, num_class):

        x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
        x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

        x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
        x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

        x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
        x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
        x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

        x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
        x = tflearn.dropout(x, 0.5, name='dropout1')

        x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
        x = tflearn.dropout(x, 0.5, name='dropout2')

        x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                    restore=False)
        return x
#     @staticmethod
#     def rnnpart(x, n_steps, n_hidden, weights, biases):
#         """ Prepare data shape to match rnn function requirements
#         current data input shape: (batch_size*nstep, n_inputs)
#         """
#         print(x.get_shape())
# # reshape input data
#         x = tf.reshape(x, [-1, n_steps, 4096])
# # permute the input data
#         x = tf.transpose(x, [1, 0, 2])
# # reshape input data
#         x = tf.reshape(x, [-1, 4096])
# # Split to get a list of 'n_step' tensors of shape (batch_size, n_step)
#         x = tf.split(0, n_steps, x)
# # Define a lstm cell with tensorflow
#         lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
# # Get lstm cell output
#         outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
# # initialize output weights
# # Linear activation, using rnn inner loop last output
#         return tf.matmul(outputs[-1], weights) + biases





