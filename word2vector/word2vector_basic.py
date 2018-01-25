
import tensorflow as tf
import numpy as np
import math
from word2vector import word2vecor_input
from word2vector.word2vecor_input import VOCABULARY_SIZE

batch_size = 256
EMBEDDING_SIZE = 128
num_negative_samples = 64
validate_size = 16
validate_data = [0]*16
for i in range (validate_size):
    validate_data[i] = 5+i*17

def train():
    w2v_input = word2vecor_input.Word2vector_input()
    with tf.Graph().as_default():
        train_inputs = tf.placeholder(tf.int32, [batch_size])
        train_labels = tf.placeholder(tf.int32, [batch_size, 1])
        validate_inputs = tf.constant(validate_data, dtype=tf.int32)

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

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
        norm_embeddings = embeddings / norm
        validata_embed = tf.nn.embedding_lookup(norm_embeddings, validate_inputs)
        similarity = tf.matmul(validata_embed, norm_embeddings, transpose_b=True)


        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            average_loss = 0
            period_test = 1000
            for i in range(100001):
                batch_input, batch_labels = w2v_input.next_batch(batch_size, 2)
                _, L = sess.run([optimizer, loss], feed_dict=
                                   {
                                       train_inputs : batch_input,
                                       train_labels : batch_labels
                                   })
                average_loss += L
                if(i % period_test ==0):
                    print('step %d, average loss =%.4f' % (i, average_loss/period_test))
                    average_loss = 0

                if(i % 10000 ==0 ):
                    sim = sess.run(similarity)
                    '''
                    sim[i, j] is the ith validate_word 's similarity to jth vocabulary word
                '''
                    print('------------------------- @step %s --------------------------' % i)
                    #sub_embeddings = tf.slice(norm_embeddings, [0, 0], [15, EMBEDDING_SIZE])
                    #print(sub_embeddings.eval())
                    for i in range(validate_size):
                        validate_word = w2v_input.map_index_to_word(validate_data[i])
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % validate_word
                        for k in range(top_k):
                            close_word = w2v_input.map_index_to_word(nearest[k])
                            log_str += close_word+','
                        print (log_str)

if __name__ == '__main__':
    train()