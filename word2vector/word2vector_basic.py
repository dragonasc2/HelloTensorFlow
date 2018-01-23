
import tensorflow as tf
import numpy as np
import math
from word2vector import word2vecor_input
from word2vector.word2vecor_input import VOCABULARY_SIZE

batch_size = 256
EMBEDDING_SIZE = 128
num_negative_samples = 64

def train():
    with tf.Graph().as_default():
        train_inputs = tf.placeholder(tf.int32, [batch_size])
        train_labels = tf.placeholder(tf.int32, [batch_size, 1])

        embeddings = tf.Variable(
            tf.truncated_normal(shape=[VOCABULARY_SIZE, EMBEDDING_SIZE], stddev=1)
        )
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weight = tf.Variable(
            tf.truncated_normal(shape=[VOCABULARY_SIZE, EMBEDDING_SIZE], stddev=1.0/math.sqrt(EMBEDDING_SIZE))
        )
        nce_biases = tf.Variable(
            tf.zeros([VOCABULARY_SIZE])
        )

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weight,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_negative_samples,
                num_classes=VOCABULARY_SIZE
            )
        )
        optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)
        input = word2vecor_input.word2vector_input()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            average_loss = 0
            period_test = 100
            for i in range(100000):
                batch_input, batch_labels = input.next_batch(batch_size, 4)
                _, L = sess.run([optimizer, loss], feed_dict=
                                   {
                                       train_inputs : batch_input,
                                       train_labels : batch_labels
                                   })
                average_loss += L
                if(i % period_test ==0):
                    print('step %d, average loss =%.4f' % (i, average_loss/period_test))
                    average_loss = 0

if __name__ == '__main__':
    train()