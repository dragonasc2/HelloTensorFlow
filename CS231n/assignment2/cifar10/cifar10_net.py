import tensorflow as tf
import numpy as np


def get_dtype():
    return tf.float32


def residual_layer(input, conv_shape, wd, is_training):
    '''
    conv-relu-conv-relu-add(input)
    :param input: 4D tensor of image features, size[N, H, W, C]
    :param conv_shape: shape of conv, list, [KH, KW, IC, OC]
    :param wd: weight decay.
    :param is_training: scalar tensor, bool.
    :return: output image features, size[N, H, W, OC]
    '''
    with tf.variable_scope('conv1'):
        weights = tf.get_variable('weights', shape=conv_shape, dtype=get_dtype(),
                                  initializer=tf.truncated_normal_initializer(stddev=1e-2))
        if wd and wd > 0:
            tf.add_to_collection('losses', wd * tf.nn.l2_loss(weights))
        biases = tf.get_variable('biases', shape=conv_shape[3], dtype=get_dtype(),
                                 initializer=tf.constant_initializer(value=1e-3))
        if(conv_shape[2] == conv_shape[3]):
            h1 = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME') + biases
        elif(conv_shape[2] * 2 == conv_shape[3]):
            # This is the case when feature size /= 2, and
            h1 = tf.nn.conv2d(input, weights, [1, 2, 2, 1], padding='SAME') + biases
        else:
            raise ValueError('conv shape %s is not allowed' % conv_shape)
        h1_bn = tf.layers.batch_normalization(h1, training=is_training)
        h1_relu = tf.nn.relu((h1_bn))
    if wd and wd>0:
        tf.add_to_collection('losses', wd * tf.nn.l2_loss(weights))

    with tf.variable_scope('conv2'):
        weights = tf.get_variable('weights', shape=[conv_shape[0], conv_shape[1], conv_shape[3], conv_shape[3]], dtype=get_dtype(),
                                  initializer=tf.truncated_normal_initializer(stddev=1e-2))
        if wd and wd > 0:
            tf.add_to_collection('losses', wd * tf.nn.l2_loss(weights))
        biases = tf.get_variable('biases', shape=conv_shape[3], dtype=get_dtype(),
                                 initializer=tf.constant_initializer(value=1e-3))
        h2 = tf.nn.conv2d(h1_relu, weights, [1, 1, 1, 1], padding='SAME') + biases
        h2_bn = tf.layers.batch_normalization(h2, training=is_training)
        h2_relu = tf.nn.relu(h2_bn)
    if(conv_shape[2] == conv_shape[3]):
        residual_added = input + h2_relu
    else:
        with tf.variable_scope('dimension_reduction'):
            weights = tf.get_variable('weights', shape=[1, 1, conv_shape[2], conv_shape[3]],
                                      dtype=get_dtype(),
                                      initializer=tf.truncated_normal_initializer(stddev=1e-2))
            if wd and wd > 0:
                tf.add_to_collection('losses', wd * tf.nn.l2_loss(weights))
            biases = tf.get_variable('biases', shape=conv_shape[3], dtype=get_dtype(),
                                     initializer=tf.constant_initializer(value=1e-3))
            input_dim_reducted = tf.nn.relu(tf.nn.conv2d(input, weights, [1, 2, 2, 1], padding='SAME') + biases)
            residual_added = input_dim_reducted + h2_relu
    return residual_added

def inference(images, is_training, wd):
    """
    cifar10 net
    :param images: 4D tensor of images. size [N, 32, 32, 3]
    :param is_training: scalar tensor, bool
    :param wd: weight decay.
    :return: logits: 1D tensor of logits, size[N]
    """
    with tf.variable_scope('conv1'):
        weights = tf.get_variable('weights', shape=[3, 3, 3, 16], dtype=get_dtype(),
                                  initializer=tf.truncated_normal_initializer(stddev=1e-2))
        if wd and wd > 0:
            tf.add_to_collection('losses', wd * tf.nn.l2_loss(weights))
        biases = tf.get_variable('biases', shape=[16], dtype=get_dtype(),
                                 initializer=tf.constant_initializer(value=1e-1))
        h1 = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME') + biases
        h1_bn = tf.layers.batch_normalization(h1)
        h1_relu = tf.nn.relu(h1_bn)

    with tf.variable_scope('size_32_32'):
        with tf.variable_scope('res1'):
            h3_1 = residual_layer(h1_relu, [3, 3, 16, 16], wd, is_training)
        with tf.variable_scope('res2'):
            h3_2 = residual_layer(h3_1, [3, 3, 16, 16], wd, is_training)
        with tf.variable_scope('res3'):
            h3_3 = residual_layer(h3_2, [3, 3, 16, 16], wd, is_training)
        with tf.variable_scope('res4'):
            h3_4 = residual_layer(h3_3, [3, 3, 16, 16], wd, is_training)
        with tf.variable_scope('res5'):
            h3_5 = residual_layer(h3_4, [3, 3, 16, 16], wd, is_training)

    with tf.variable_scope('size_16_16'):
        with tf.variable_scope('res1'):
            h4_1 = residual_layer(h3_5, [3, 3, 16, 32], wd, is_training)
        with tf.variable_scope('res2'):
            h4_2 = residual_layer(h4_1, [3, 3, 32, 32], wd, is_training)
        with tf.variable_scope('res3'):
            h4_3 = residual_layer(h4_2, [3, 3, 32, 32], wd, is_training)
        with tf.variable_scope('res4'):
            h4_4 = residual_layer(h4_3, [3, 3, 32, 32], wd, is_training)
        with tf.variable_scope('res5'):
            h4_5 = residual_layer(h4_4, [3, 3, 32, 32], wd, is_training)

    with tf.variable_scope('size_8_8'):
        with tf.variable_scope('res1'):
            h5_1 = residual_layer(h4_5, [3, 3, 32, 64], wd, is_training)
        with tf.variable_scope('res2'):
            h5_2 = residual_layer(h5_1, [3, 3, 64, 64], wd, is_training)
        with tf.variable_scope('res3'):
            h5_3 = residual_layer(h5_2, [3, 3, 64, 64], wd, is_training)
        with tf.variable_scope('res4'):
            h5_4 = residual_layer(h5_3, [3, 3, 64, 64], wd, is_training)
        with tf.variable_scope('res5'):
            h5_5 = residual_layer(h5_4, [3, 3, 64, 64], wd, is_training)

    h6_flat = tf.reshape(h5_5, [-1, 8 * 8 * 64])
    with tf.variable_scope('FC1'):
        
        h6_dropout = tf.cond(is_training, lambda: tf.nn.dropout(h6_flat, keep_prob=0.5), lambda: tf.nn.dropout(h6_flat, keep_prob=1))
        weights_shape = [8 * 8 * 64, 1024]
        weights = tf.get_variable('weights', shape=weights_shape, dtype=get_dtype(),
                                  initializer=tf.contrib.layers.xavier_initializer(weights_shape))
        biases = tf.get_variable('biases', shape=[1024], dtype=get_dtype(),
                                 initializer=tf.constant_initializer(value=0))
        fc1 = tf.nn.relu(tf.matmul(h6_dropout, weights) + biases)
        fc1_dropout = tf.cond(is_training, lambda: tf.nn.dropout(fc1, keep_prob=0.5), lambda: tf.nn.dropout(fc1, keep_prob=1))



    with tf.variable_scope('softmax'):
        weights_shape = [1024, 10]
        weights = tf.get_variable('weights', shape=weights_shape, dtype=get_dtype(),
                                  initializer=tf.contrib.layers.xavier_initializer(weights_shape))
        if wd and wd > 0:
            tf.add_to_collection('losses', wd * tf.nn.l2_loss(weights))
        biases = tf.get_variable('biases', shape=[10], dtype=get_dtype(),
                                 initializer=tf.constant_initializer(value=0))
        logits = tf.nn.softmax(tf.matmul(fc1_dropout, weights) + biases)

    return logits


def loss(logits, labels):
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), 'total_loss')

