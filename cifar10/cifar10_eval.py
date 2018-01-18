import math
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from cifar10 import cifar10_cnn


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval', 'Directory to write event logs')
tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train', ' Directory to read model checkpoints')
tf.app.flags.DEFINE_integer('num_examples', 10000, 'Number of examples to run')


def do_eval():
    images, labels = cifar10_cnn.inputs(True)
    logits = cifar10_cnn.inference(images)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10_cnn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt)
        else:
            raise ValueError('No checkpoint file found')

        num_iter = int(FLAGS.num_examples / FLAGS.batch_size)
        true_count = 0
        total_sample_count = 0
        step = 0
        while step<num_iter:
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            total_sample_count += FLAGS.batch_size
            step += 1

        precision = true_count / total_sample_count
        print('%s : precision %.3f' % (datetime.now(), precision))

if __name__ == '__main__':
    do_eval()