#from __future__ import absolute_import
#from __future__ import division

from _datetime import datetime
import time
import tensorflow as tf

from cifar10 import cifar10_cnn
from cifar10 import cifar10_eval

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train',
                           'Diectory where to write event logs and checkpoint')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'number of batches to run')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'How often to log results to the console')



def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        images, labels = cifar10_cnn.distorted_inputs()

        #with tf.device('/device:GPU:0'):
        logits = cifar10_cnn.inference(images)

        loss = cifar10_cnn.loss(logits, labels)

        train_op, learning_rate = cifar10_cnn.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs([loss, learning_rate])

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    [loss_value, lr]= run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration) / FLAGS.log_frequency

                    format_str = '%s : step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch, learning rate:%.6f)'
                    print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch, lr))
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_secs=None,
            hooks=[
                tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                #tf.train.StopAtStepHook(last_step=200),
                tf.train.NanTensorHook(loss),
                _LoggerHook()
            ],
            config=config

        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    cifar10_cnn.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()














