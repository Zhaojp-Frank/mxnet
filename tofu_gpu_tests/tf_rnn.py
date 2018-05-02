import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import time

def RNN(x, n_layers=6, n_hidden=8192, n_seq=20, n_class=512):
    x = tf.split(x, n_seq, 1)
    weights = None
    for i in range(n_layers):
        with tf.device('/device:GPU:{}'.format(i)):
            with tf.variable_scope('lstm_layer_{}'.format(i)) as scope:
                rnn_cell = rnn.BasicLSTMCell(n_hidden)
                if weights is None:
                    weights = rnn_cell.trainable_weights
                outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
                x = outputs
    outputs = [layers.fully_connected(o, n_class, biases_initializer=None)
                for o in outputs]
    outputs = [layers.softmax(o) for o in outputs]
    # return tf.group(*outputs)
    return tf.add_n(outputs), weights


def run(n_batch=32, n_layers=6, n_seq=20, n_embed=512, n_class=512):
    with tf.device('/device:GPU:0'):
        data = tf.placeholder(tf.float32, (n_batch, n_embed * n_seq))
        loss, weights = RNN(data, n_layers=n_layers, n_seq=n_seq, n_class=n_class)
        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        # train_op = optimizer.minimize(loss)
        train_op = tf.gradients(loss, weights,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    x = np.random.rand(n_batch, n_embed * n_seq)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('-------------------')
    for i in range(10):
        begin = time.time()
        sess.run(train_op, feed_dict={data:x})
        end = time.time()
        print(end - begin)


if __name__ == '__main__':
    run(n_batch=128,
        n_layers=1,
        n_seq=20,
        n_embed=512,
        n_class=512)
