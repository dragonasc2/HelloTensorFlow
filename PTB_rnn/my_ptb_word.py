# Author : Dragon_n

import time
import numpy as np
import tensorflow as tf

from PTB_rnn import reader
from PTB_rnn import util

tf.flags.DEFINE_string(
    "model", "small",
    "Type of Model. Options are : small, medium, large"
)
tf.flags.DEFINE_string(
    "data_path", "PTB_data",
    "where to store the training/testing data"
)
tf.flags.DEFINE_string(
    "save_path", None, #"my_log_save",
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
tf.flags.DEFINE_string(
    "rnn_mode", "basic",
    "the low level implementation of lstm cell : CUDNN OR BASIC OR BLOCK"
)
FLAGS = tf.flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

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
        self._rnn_params = None
        self._cell = None
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

        output, state = self._build_rnn_graph(inputs, config, is_training)

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

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                           config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step()
        )
        self._new_lr = tf.placeholder(
            tf.float32,
            name='new_learning_rate'
        )
        self._lr_update_op = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update_op, feed_dict={self._new_lr: lr_value})
        
    def _build_rnn_graph(self, inputs, config, is_training):
        if(config.rnn_mode == CUDNN):
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)
        

    def export_ops(self, name):
        """
        Exports ops to collections
        :param name:
        :return:
        """
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_traning:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update_op=self._lr_update_op)
            if self._rnn_params:
                ops.update(run_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        if self._is_traning:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update_op = tf.get_collection_ref("lr_update_op")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN"
                )
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas
        )
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas
        )

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        inputs = tf.transpose(inputs, [1, 0, 2])
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

    def _get_lstm_cell(self, config, is_training):
        if(config.rnn_mode == BASIC):
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training
            )
        if(config.rnn_mode == BLOCK):
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0
            )
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)
    
    
    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob
                )
            return cell
        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True
        )
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

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

def run_epoch(session, model, eval_op=None, verbose=False):
    """
    runs the model on the given data
    :param session:
    :param model:
    :param eval_op:
    :param verbose:
    :return:
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step %(model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) / (time.time() - start_time)))

    return np.exp(costs  / iters)

def get_config():
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig
    else:
        raise ValueError("Invalid model: %s" % FLAGS.model)
    config.rnn_mode = FLAGS.rnn_mode
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale,
            config.init_scale
        )

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", train_model.cost)
            tf.summary.scalar("Learning Rate", train_model.lr)


        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True):
                valid_model = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Valid Loss", valid_model.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = PTBModel(is_training=False, config=eval_config, input_=test_input)
            tf.summary.scalar("Test Loss", test_model.cost)

        models = {"Train": train_model} #, "Valid": valid_model, "Test":test_model}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        #util.auto_parallel(metagraph, model)
        with tf.Graph().as_default():
            tf.train.import_meta_graph(metagraph)
            for model in models.values():
                model.import_ops()
            sv = tf.train.Supervisor(logdir=FLAGS.save_path)
            config_proto = tf.ConfigProto(allow_soft_placement=False)
            with sv.managed_session(config=config_proto) as sess:
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    train_model.assign_lr(sess, config.learning_rate * lr_decay)
                    print("epoch: %d Leanring Rate: %.3f" % (i, sess.run(train_model.lr)))

                    train_perplexity = run_epoch(sess, train_model, eval_op=train_model.train_op, verbose=True)

                    print("epoch: %d, valid Perplexity: %.3f" % ((i+1), run_epoch(sess,valid_model)))

                print("Test Perplexity: %.3f" % run_epoch(sess, test_model))


if __name__ == '__main__':
    tf.app.run();








