
'''
Re-implementation of Heat Diffusion Long Short-Term Memory with TensorFlow
'''
from __future__ import print_function
import scipy.io as spio
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from dataset_retrieval import Dataset
import numpy as np
import pickle
import numpy.matlib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from dataset_retrieval import loadmat, _check_keys, _todict
# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 64
display_step = 500

test_step = 2000

# Network Parameters
n_input = 128 # histogram dimension
n_steps = 101 # timesteps
n_hidden = 100 # hidden layer num of features
n_output = 100
n_classes = 6 # 3D shape classes



# define tf placeholders
x = tf.placeholder("float", [n_steps, n_input, None])
y = tf.placeholder("float", [None, n_output])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}



def HD_LSTM(x, weights, biases):


    x = tf.transpose(x, [0, 2, 1])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Target_vectors = np.random.rand(n_classes,n_hidden)
mat_content = loadmat('Target_vectors_random.mat')
Target_vectors = np.array(mat_content['Target_vectors'])

pred = HD_LSTM(x, weights, biases)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))




cost = tf.nn.l2_loss((pred-y), name=None)/(batch_size*batch_size)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

dataset = Dataset(n_classes=n_classes, train_path='ShapeNet_hist_train_6/', test_path='ShapeNet_hist_test_6/', shuffleType='normal', seqLength=0, target=Target_vectors, n_hidden=n_output)

plot_iter = np.zeros((training_iters, 1))
plot_training_loss = np.zeros((training_iters, 1))

with tf.Session() as sess:
    sess.run(init)
    step = 1
    plot_count = 0
    all_loss = np.zeros((training_iters, 1))

    while step < training_iters:

        batch_data, batch_target_vector = dataset.next_batch(batch_size, 'train')

        batch_x = batch_data
        batch_y = batch_target_vector

        # print(len(label_data))
        # batch_y = np.zeros((len(label_data), n_hidden))
        # for i in range(len(label_data)):
        #     batch_y[i,:] = Target_vectors[label_daata[i],:]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # Display testing status
        if step%test_step == 0:
            test_cost = 0.
            test_count = 0
            for _ in range(int(dataset.test_size/batch_size)):
                batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
  
                temp_cost = sess.run(cost, feed_dict={x: batch_tx, y: batch_ty})
                test_cost += temp_cost
                test_count += 1
            test_cost /= test_count
            print("Iter " + str(step) + ", Testing loss = " + "{:.5f}".format(test_cost))


            test_data, test_label = dataset.final_eval('test')
            train_data, train_label = dataset.final_eval('train')
            #train_data_temp2, train_label_value = dataset.final_eval_label('train')


            test_feature = sess.run(pred, feed_dict={x: test_data})
            train_feature = sess.run(pred, feed_dict={x: train_data})
            # train_feature = np.zeros((len(train_label_value), n_hidden))
            # for i in range(len(train_label_value)):
            #     train_feature[i,:] = Target_vectors[train_label_value.astype(int)[i]-1,:]


            precision = dict()
            recall = dict()
            average_precision = 0
            thresholds = dict()
            for i in range(len(test_label)):
                # repmat_test_data = np.matlib.repmat(test_feature[i,:], len(train_label) , 1)
                y_scores = np.zeros((len(train_label), 1))
                for j in range(len(train_label)):
                    y_scores[j] = 1/(np.linalg.norm(test_feature[i,:]-train_feature[j,:])+0.00001)

                label_index = train_label[:, test_label.astype(int)[i]]

                # y_true = np.array([0, 0, 1, 1])
                # y_scores = np.array([0.1, 0.4, 0.35, 0.8])

                precision[i], recall[i], thresholds[i] = precision_recall_curve( np.array(train_label[:,test_label.astype(int)[i]]), np.array(y_scores) )

                average_precision  += average_precision_score( np.array(train_label[:,test_label.astype(int)[i]]), np.array(y_scores) )

            mAP = average_precision/len(test_label)
            print(average_precision/len(test_label))
            # if mAP > 0.48:
            #     step = training_iters+1


        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
            plot_iter[plot_count] = step
            plot_training_loss[plot_count] = loss
            all_loss[plot_count]=loss
            plot_count += 1
        step += 1
    print("Optimization Finished!")

    # # Calculate accuracy for the query data (Only use a simple recognition task
    # for evaluating the learned shape representation)
    test_cost = 0.
    test_count = 0
    for _ in range(int(dataset.test_size/batch_size)):
        batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')

        temp_cost = sess.run(cost, feed_dict={x: batch_tx, y: batch_ty})
        test_cost += temp_cost
        test_count += 1
    test_cost /= test_count
    print("Testing cost = " + "{:.5f}".format(test_cost))

    test_data, test_label = dataset.final_eval('test')
    train_data, train_label = dataset.final_eval('train')
    # train_data_temp2, train_label_value = dataset.final_eval_label('train')

    test_feature = sess.run(pred, feed_dict={x: test_data})
    train_feature = sess.run(pred, feed_dict={x: train_data})
    # train_feature = np.zeros((len(train_label_value), n_output))
    # for i in range(len(train_label_value)):
    #     train_feature[i,:] = Target_vectors[train_label_value.astype(int)[i]-1,:]


    precision = dict()
    recall = dict()
    average_precision = 0
    thresholds = dict()
    for i in range(len(test_label)):
        # repmat_test_data = np.matlib.repmat(test_feature[i,:], len(train_label) , 1)
        y_scores = np.zeros((len(train_label), 1))
        for j in range(len(train_label)):
            y_scores[j] = 1/(np.linalg.norm(test_feature[i,:]-train_feature[j,:])+0.00001)

        label_index = train_label[:, test_label.astype(int)[i]]

        # y_true = np.array([0, 0, 1, 1])
        # y_scores = np.array([0.1, 0.4, 0.35, 0.8])

        precision[i], recall[i], thresholds[i] = precision_recall_curve( np.array(train_label[:,test_label.astype(int)[i]]), np.array(y_scores) )

        average_precision  += average_precision_score( np.array(train_label[:,test_label.astype(int)[i]]), np.array(y_scores) )


    print(average_precision/len(test_label))

    # pickle.dump( precision, open( "precision_CCA.txt", "wb" ) )
    # pickle.dump( recall, open( "recall_CCA.txt", "wb" ) )
    pickle.dump(all_loss, open("batch_loss_LSTM.txt", "wb"))
    # pickle.dump( plot_training_loss, open( "plot_training_loss_CCA.txt", "wb" ) )
    # pickle.dump( plot_iter, open( "plot_iter_CCA.txt", "wb" ) )
    # pickle.dump( train_feature, open( "train_feature_random.txt", "wb" ) )
    # pickle.dump( train_label, open( "train_label_random.txt", "wb" ) )

    # plt.plot(plot_iter[:plot_count-1], plot_training_loss[:plot_count-1], 'ro')
    # plt.ylabel('Training error')
    # plt.show()
    # savefig('figure.pdf')

# the higher the dimension of hidden layer is set to be, the better the performance is