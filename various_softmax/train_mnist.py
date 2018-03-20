import tensorflow as tf
import input_data
import numpy as np
from various_softmax import a_softmax_loss
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

NUM_CLASS = 10
IMAGE_H = 28
IMAGE_W = 28
BATCH_SIZE = 128


def inference(image_input, embedding_size):
    net = tf.layers.conv2d(image_input, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
    #net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='SAME')
    net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu)

    net = tf.layers.conv2d(net, filters=64, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
    #net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='SAME')
    net = tf.layers.conv2d(net, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu)

    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
    #net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='SAME')
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu)


    net = tf.layers.flatten(net)
    embeddings = tf.layers.dense(net, embedding_size)
    return embeddings


def cal_logits(embeddings, labels, num_class, m):
    logits, loss = a_softmax_loss.a_softmax_loss(embeddings, labels, num_class, m, 'various_softmax')
    # logits = tf.layers.dense(embeddings, num_class)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return logits, loss

def visual_feature_space(features, labels, num_class):
    num = len(labels)
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    class_color_map = dict([(i, [i / num_class, 0.5, 1 - i / num_class]) for i in range(num_class)])
    feature_color = [class_color_map[labels[i]] for i in range(num)]
    sc = ax.scatter(features[:, 0], features[:, 1], c=feature_color, lw=0, s=10)
    for i in range(num_class):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, '%s' % (i, ))
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
    plt.show()


def do_eval(sess,
            eval_correct,
            image_flat_placeholder,
            labels_one_hot_placeholder,
            data_set
            ):
    total_steps = data_set.num_examples // BATCH_SIZE
    sample_count = 0
    true_count = 0

    for i in range(total_steps):
        img, lbl = data_set.next_batch(BATCH_SIZE)
        true_count += BATCH_SIZE * sess.run(eval_correct,feed_dict=
        {
            image_flat_placeholder : img,
            labels_one_hot_placeholder : lbl
        })
        sample_count += BATCH_SIZE
    return true_count/sample_count


def main():
    data_set = input_data.read_data_sets('/MNIST_data', False, True)
    image_flat_placeholder = tf.placeholder(tf.float32, [None, IMAGE_H * IMAGE_W])
    image = tf.reshape(image_flat_placeholder, [-1, IMAGE_H, IMAGE_W, 1])
    labels_one_hot_placeholder = tf.placeholder(tf.float32, [None, NUM_CLASS])
    labels = tf.argmax(labels_one_hot_placeholder, 1)
    keep_prob = tf.placeholder(tf.float32)
    embeddings = inference(image, 2)
    logits, loss = cal_logits(embeddings, labels, NUM_CLASS, m=1)

    learning_rate = 5e-4
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    eval_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20001):
            batch_images_flat, batch_labels = data_set.train.next_batch(BATCH_SIZE)
            _, this_train_loss = sess.run([train_step, loss], feed_dict={
                image_flat_placeholder: batch_images_flat,
                labels_one_hot_placeholder: batch_labels
            })
            if i % 100 == 0:
                print('%dth iter: train loss : %.4f' % (i, this_train_loss))
                print('test accuracy : %.4f' % (do_eval(sess, eval_correct, image_flat_placeholder,
                                                            labels_one_hot_placeholder, data_set.test)))
            if i % 5000 == 0:
                sample_data = np.array(range(data_set.test.images.shape[0]))
                np.random.shuffle(sample_data)
                #sample_data = sample_data[:BATCH_SIZE]
                test_data = data_set.test.images[sample_data, :]
                test_labels = data_set.test.labels[sample_data]
                feat_vec = sess.run(embeddings, feed_dict={image_flat_placeholder: test_data})
                visual_feature_space(feat_vec, np.argmax(test_labels, 1), NUM_CLASS)






if __name__ == '__main__':
    main()