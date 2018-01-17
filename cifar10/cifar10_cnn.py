import os
import sys
from six.moves import urllib
import tensorflow as tf
import tarfile
import re
import cifar10.cifar10_input as cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size for training')
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data', 'the directory of training/test data of cifar10')
tf.app.flags.DEFINE_bool('use_fp16', False,' Train the model with float16')
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored in CPU memory
    :param name: name of the variable
    :param shape:  list of ints
    :param initializer: initiallizer for Variables
    :return: Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        tf.get_variable()
        var = tf.Variable(shape=shape, initial_value=initializer, name=name,dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create and initialized Variable with weight decay
    :param name: name of the variable
    :param shape: list of ints
    :param stddev: standard deriation of a truncate Gaussian
    :param wd: add L2Loss weight decay multiply by this float.
    :return: Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal(shape, stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    return var


def distorted_inputs():
    """
    Construct distored input for CIFAR training using the Reader ops.

    :return:
        images : Images, 4D tensor for [batch_size, Image_size, Image_size, 3 ]
        labels : Labels. 1D tensor of [batch_size

    Raises:
        ValueError: if no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distored_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images,labels


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data, data_dir, FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images,labels


def inference(images):
    """
    build the CIFAR-10 model
    :param images: Images returned from distored_inputs() or inputs(). 4D tensor
    :return: logits
    """
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([5, 5, 3, 64],stddev=0.1)
        )
        biases = tf.Variable(
            tf.constant(0.01, dtype=tf.float32, shape=[64])
        )
        h_conv1 = tf.nn.relu(tf.nn.conv2d(images, weights, [1,1,1,1],'SAME') + biases, name=scope.name)
        _activation_summary(h_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool')
        h_norm1 = tf.nn.lrn(h_pool1, depth_radius=5, bias=1.0, alpha= 0.001 / 9, beta=0.75, name='norm')

    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(
            tf.truncated_normal([5, 5, 64, 64], stddev=0.1),
            name='weights'
        )
        biases = tf.Variable(
            tf.constant(0.01, tf.float32, [64]),
            name='biases'
        )
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1, weights, [1, 1, 1, 1], 'SAME'))
        _activation_summary(h_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1],[1, 2, 2, 1], 'SAME', name='pool')
        h_norm2 = tf.nn.lrn(h_pool2, depth_radius=5, bias=1.0, alpha = 0.001 / 9, beta=0.75, name='norm')

    h_norm2_flat = tf.reshape(h_norm2, [-1, h_norm2.shape[1]*h_norm2.shape[2]*h_norm2.shape[3]])
    with tf.variable_scope('local3') as scope:
        weights = tf.Variable(
            tf.truncated_normal([h_norm2_flat.get_shape()[1].value, 384], stddev=0.1),
            name='weights'
        )
        biases = tf.Variable(
            tf.constant(0.01,dtype=tf.float32,shape=[384]),
            name='biase'
        )
        h_local3 = tf.nn.relu(tf.matmul(h_norm2_flat, weights) + biases)
        _activation_summary(h_local3)

    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(
            tf.truncated_normal([384, 192], stddev=0.1),
            name='weights'
        )
        biases = tf.Variable(
            tf.constant(0.01,dtype=tf.float32,shape=[192])
        )
        h_local4 = tf.nn.relu(tf.matmul(h_local3, weights) + biases)
        _activation_summary(h_local4)

    with tf.variable_scope('linear_softmax') as scope:
        weights = tf.Variable(
            tf.truncated_normal([192, NUM_CLASSES], stddev=0.1),
            name='weights'
        )
        biases = tf.Variable(
            tf.constant(0.01, dtype=tf.float32, shape=[NUM_CLASSES]),
            name='biases'
        )
        logits = tf.nn.softmax(tf.matmul(h_local4, weights) + biases)
        #_activation_summary(logits)

    return logits

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels), name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)',l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(
        learning_rate=INITIAL_LEARNING_RATE,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=LEARNING_RATE_DECAY_FACTOR
    )
    tf.summary.scalar('learning_rate', lr)
    loss_average_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


def maybe_download_and_extract():
    dst_dir = FLAGS.data_dir
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    file_name = DATA_URL.split('/')[-1]
    file_path = os.path.join(dst_dir, file_name)
    if not os.path.exists(file_path):
        def _progress(block_num, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (file_name, block_num*block_size/total_size*100))
            sys.stdout.flush()
        downloaded_file_path, _ = urllib.request.urlretrieve(DATA_URL,file_path,_progress)
        print()
        statinfo = os.stat(downloaded_file_path)
        print('Downloaded ', file_name, statinfo.st_size, 'bytes')
    extract_dir_path = os.path.join(dst_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extract_dir_path):
        tarfile.open(file_path,'r:gz').extractall(dst_dir)



def main():
    maybe_download_and_extract()


if __name__ == '__main__':
    main()















