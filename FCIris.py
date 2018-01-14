
import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

work_dir = "Iris_data"

# Data sets
IRIS_TRAINING = os.path.join(work_dir,"iris_training.csv")
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = os.path.join(work_dir,"iris_test.csv")
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

NUM_CLASS = 3


def main():
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL)
        with open(IRIS_TRAINING) as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL)
        with open(IRIS_TEST) as f:
            f.write(raw)

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TRAINING,
        target_dtype = np.int,
        features_dtype = np.float32
    )
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TEST,
        target_dtype = np.int,
        features_dtype = np.float32
    )
    x_placeholder = tf.placeholder(tf.float32,[None,4])
    labels_placeholder = tf.placeholder(tf.int32,[None])

    with tf.name_scope("fc1"):
        fc1_weights = tf.Variable(
            tf.truncated_normal([4,10],stddev = 0.1))
        fc1_biases = tf.Variable(
            tf.constant(0.1,tf.float32,[10])
        )
        h_fc1 = tf.nn.relu(tf.matmul(x_placeholder,fc1_weights)+fc1_biases)

    with tf.name_scope("fc2"):
        fc2_weights = tf.Variable(
            tf.truncated_normal([10,20],stddev = 0.1)
        )
        fc2_biases = tf.Variable(
            tf.constant(0.1,tf.float32,[20])
        )
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,fc2_weights)+fc2_biases)

    with tf.name_scope("fc3"):
        fc3_weights = tf.Variable(
            tf.truncated_normal([20,10],stddev = 0.1)
        )
        fc3_biases = tf.Variable(
            tf.constant(0.1,tf.float32,[10])
        )
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2,fc3_weights)+fc3_biases)

    with tf.name_scope("softmax"):
        fc4_weights = tf.Variable(
            tf.truncated_normal([10,3],stddev = 0.1)
        )
        fc4_biases = tf.Variable(
            tf.constant(0.1,tf.float32,[3])
        )
        y = tf.nn.softmax(tf.matmul(h_fc3,fc4_weights)+fc4_biases)

    labels = tf.to_int64(labels_placeholder)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=labels,name = 'cross_entropy')
    L = tf.reduce_mean(cross_entropy)

    eval_correct = tf.nn.in_top_k(y,labels,1)
    accuracy = tf.reduce_mean(tf.cast(eval_correct,tf.float32))

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(L)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            x_batch = training_set.data
            labels_batch = training_set.target
            sess.run(optimizer,feed_dict=
            {
                x_placeholder : x_batch,
                labels_placeholder : labels_batch
            })

            if(i%100 ==0):
                x_batch = test_set.data
                labels_batch = test_set.target

                print('step %d : accuracy %.4f' % (i,sess.run(accuracy,
                                                              feed_dict=
                                                              {
                                                                  x_placeholder : x_batch,
                                                                  labels_placeholder : labels_batch
                                                              })))






if __name__ =="__main__":
    main();













