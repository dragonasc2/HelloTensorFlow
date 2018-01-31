# Author : Dragon_n

import time
import numpy as np
import tensorflow as tf

from PTB_rnn import reader
from PTB_rnn import util

tf.flags.DEFINE_String(
    "model", "small",
    "Type of Model. Options are : small, medium, large"
)
tf.flags.DEFINE_string(
    "data_path", "PTB_data",
    "where to store the training/testing data"
)
tf.flags.DEFINE_String(
    "save_path", None,
    "Model output directgory"
)
tf.flags.DEFINE_bool(
    "use_fp16", False,
    "use 16bits float or not"
)
tf.flags.DEFINE_integer(
    "num_gpus", 1,
    "number of GPU. It's useless now"
)
FLAGS = tf.flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """
    The PTB input data
    """
    def __init__(self, config, data, name=None):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.epoch_size = ((len(data) // self.batch_size)-1) // self.num_steps
        self.input_data, self.targets = reader.ptb_producer(data, self.batch_size, self.num_steps)


class PTBModel(object):
    """
    The PTB model
    """
    def __init__(self, is_training, config, input_):
        self._is_traning = is_training
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        embedding = tf.get_variable(
            "embedding", shape=[vocab_size, hidden_size], dtype=data_type()
        )
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph_cudnn(inputs, config, is_training)

        softmax_weights = tf.get_variable(
            "softmax_weights", [hidden_size, vocab_size], data_type()
        )
        softmax_biases = tf.get_variable(
            "softmax_biases", [vocab_size], data_type()
        )
        logits = tf.nn.xw_plus_b(output, softmax_weights, softmax_biases)
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], data_type()),
            average_across_timesteps=False,
            average_across_batch=True
        )
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        # TODO HERE



    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        inputs = tf.tranpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0
        )
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale
            ),
            validate_shape=False
        )
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size], data_type())
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size], data_type())
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

