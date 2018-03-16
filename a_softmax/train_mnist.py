import tensorflow as tf
import input_data
import numpy

NUM_CLASS = 10
IMAGE_H = 28
IMAGE_W = 28
BATCH_SIZE = 100




def inference(image_input, embedding_size):
    net = tf.layers.conv2d(image_input, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='SAME')
    net = tf.layers.conv2d(image_input, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='SAME')
    net = tf.layers.conv2d(image_input, filters=32, kernel_size=3, strides=1, padding='SAME')
    net = tf.reshape(net, shape=[-1, 7*7*32])
    embeddings = tf.layers.dense(net, embedding_size)
    return embeddings


def train():
    data_set = input_data.read_data_sets('/MNIST_data', False, True)
    image_placeholder = tf.placeholder(tf.float32, [None, IMAGE_H, IMAGE_W, 1])
    labels_placeholder = tf.placeholder(tf.float32, [None, NUM_CLASS])
    keep_prob = tf.placeholder(tf.float32)
    logits = inference(image_placeholder, 128)




if __name__ == '__main__':
    train()