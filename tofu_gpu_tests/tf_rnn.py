import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import time

def RNN(x, n_layers=6, n_hidden=8192, n_seq=20, n_class=512):
    # x = tf.split(x, n_seq, 1)
    weights = None
    for i in range(n_layers):
        with tf.device('/device:GPU:{}'.format(i % 8)):
            with tf.variable_scope('lstm_layer_{}'.format(i)) as scope:
                rnn_cell = rnn.BasicLSTMCell(n_hidden)
                if weights is None:
                    weights = rnn_cell.trainable_weights
                outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
                x = outputs
    with tf.device('/device:GPU:{}'.format(7)):
        outputs = [layers.fully_connected(o, n_class, biases_initializer=None)
                    for o in outputs]
        outputs = [layers.softmax(o) for o in outputs]
        result = tf.add_n(outputs)
    # return tf.group(*outputs)
    return result, weights


def run(n_batch=32, n_layers=6, n_seq=20, n_embed=512, n_class=512):
    data = []
    feed_dict = {}
    for i in range(n_seq):
        # name = 'data{}'.format(i)
        data.append(tf.placeholder(tf.float32, (n_batch, n_embed)))
        feed_dict[data[-1].name] = np.random.rand(n_batch, n_embed)
    loss, weights = RNN(data, n_layers=n_layers, n_seq=n_seq, n_class=n_class)
    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    # train_op = optimizer.minimize(loss)
    train_op = tf.gradients(loss, weights,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('-------------------')
    for i in range(3):
        begin = time.time()
        sess.run(train_op, feed_dict=feed_dict)
        end = time.time()
        print(end - begin)


if __name__ == '__main__':
    run(n_batch=1,
        n_layers=6,
        n_seq=20,
        n_embed=512,
        n_class=512)
