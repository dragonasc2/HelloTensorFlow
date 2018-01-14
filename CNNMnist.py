import tensorflow as tf
import input_data
import numpy

NUM_CLASS = 10
IMAGE_H = 28
IMAGE_W = 28
BATCH_SIZE = 100

def interence(images_flat,conv1_size,conv2_size,fc1_size,keep_prob):

    images = tf.reshape(images_flat,[-1,IMAGE_H,IMAGE_W,1])
    with tf.name_scope('conv1'):
        W_conv1 = tf.Variable(
            tf.truncated_normal(conv1_size,stddev = 0.1),
            name = 'weights'
        )
        b_conv1 = tf.Variable(
            tf.constant(0.1,shape=[conv1_size[3]]),
            #tf.zeros(shape=[conv1_size[3]]),
            name = 'biases'
        )
        h_conv1 = tf.nn.relu(tf.nn.conv2d(images,W_conv1,[1,1,1,1],'SAME')+b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1,[1,2,2,1],[1,2,2,1],'SAME');

    with tf.name_scope('conv2'):
        W_conv2 = tf.Variable(
            tf.truncated_normal(conv2_size,stddev=0.1),
            name = 'weights'
        );
        b_conv2 = tf.Variable(
            tf.constant(0.1,shape=[conv2_size[3]]),
            #tf.zeros(shape=[conv2_size[3]]),
            name = 'biases'
        );
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,[1,1,1,1],'SAME')+b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2,[1,2,2,1],[1,2,2,1],'SAME');

    h_pool2_flat = tf.reshape(h_pool2,[-1,IMAGE_H//4 * IMAGE_W//4 * conv2_size[3]])

    with tf.name_scope('fc1'):
        W_fc1 = tf.Variable(
            tf.truncated_normal(fc1_size,stddev=0.1),
            name = 'weights'
        );
        b_fc1 = tf.Variable(
            tf.constant(0.1,shape=[fc1_size[1]]),
            #tf.zeros(shape=[fc1_size[1]]),
            name = 'biases'
        )
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1);

    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    with tf.name_scope('linear_softmax'):
        W_fc2 = tf.Variable(
            tf.truncated_normal([fc1_size[1],NUM_CLASS],stddev=0.1),
            name = 'weights'
        );
        b_fc2 = tf.Variable(
            tf.constant(0.1,shape=[NUM_CLASS]),
            #tf.zeros(shape=[NUM_CLASS]),
            name = 'biases'
        );
        # softmax 内绝对不能有relu
        # 即使配和dropout，使得其可以训练，这样使用也是不对的。
        logits = tf.nn.softmax(
            ((tf.matmul(h_fc1_drop,W_fc2)+b_fc2))
        )
    return logits



def loss(logits,labels_placeholder):
    return -tf.reduce_sum(labels_placeholder*tf.log(logits))

def train(loss,learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            keep_prob,
            data_set
            ):
    total_steps = data_set.num_examples // BATCH_SIZE
    sample_count = 0
    true_count = 0

    for i in range(total_steps):
        img, lbl = data_set.next_batch(BATCH_SIZE)
        true_count += BATCH_SIZE * sess.run(eval_correct,feed_dict=
        {
            images_placeholder : img,
            labels_placeholder : lbl,
            keep_prob : 1
        })
        sample_count += BATCH_SIZE
    return true_count/sample_count

def main():
    data_set = input_data.read_data_sets('/MNIST_data',False,True)
    image_placeholder = tf.placeholder(tf.float32,[None,28*28])
    labels_placeholder = tf.placeholder(tf.float32,[None,NUM_CLASS])
    keep_prob = tf.placeholder(tf.float32)
    logits = interence(image_placeholder,[5,5,1,32],[5,5,32,64],[7*7*64,1024],keep_prob)
    #cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(logits))
    cross_entropy = loss(logits,labels_placeholder)
    optimizer = train(cross_entropy,1e-4)
    eval_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(labels_placeholder,1)),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            batch_images_flat,batch_labels = data_set.train.next_batch(BATCH_SIZE)
            sess.run(optimizer,feed_dict={
                image_placeholder : batch_images_flat,
                labels_placeholder : batch_labels,
                keep_prob : 1
            })
            if(i%100==0):
                print('step:%d precision on test : %f' %(i,do_eval(sess,eval_correct,image_placeholder,labels_placeholder,keep_prob,data_set.test)))

if __name__=='__main__':
    main()

















