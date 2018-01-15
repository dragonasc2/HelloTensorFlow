import tensorflow as tf
import numpy as np
import SRCNN.prepare_training



def main():
    LR_placeholder = tf.placeholder(tf.float32,[None,None,None,1],'bicubiced_LR')
    GT_placeholder = tf.placeholder(tf.float32,[None,None,None,1],'ground_truth_SR')

    with tf.name_scope("conv1"):
        weights = tf.Variable(
            tf.truncated_normal([9,9,1,64],stddev=0.001),
            name = 'weights'
        )
        biases = tf.Variable(
            tf.constant(0.00,dtype=tf.float32,shape=[64]),
            name = 'biases'
        )
        h_conv1 = tf.nn.relu(tf.nn.conv2d(LR_placeholder,weights,strides = [1,1,1,1],padding='VALID')+biases)

    with tf.name_scope("conv2"):
        weights = tf.Variable(
            tf.truncated_normal([1,1,64,32],stddev=0.001),
            name = 'weights'
        )
        biases = tf.Variable(
            tf.constant(0.00,dtype=tf.float32,shape=[32]),
            name = 'biases'
        )
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,weights,strides=[1,1,1,1],padding='VALID')+biases)

    with tf.name_scope("conv3"):
        weights = tf.Variable(
            tf.truncated_normal([5,5,32,1],stddev = 0.001),
            name = 'weights'
        )
        biases = tf.Variable(
            tf.constant(0.00,dtype=tf.float32,shape=[1]),
            name = 'biases'
        )
        SR = tf.nn.conv2d(h_conv2,weights,strides=[1,1,1,1],padding='VALID')+biases
    MSE = tf.reduce_mean(tf.square(SR-GT_placeholder))
    PSNR = 10*tf.log(1/MSE)/tf.log(10.0)

    optimizer = tf.train.AdamOptimizer(1e-5).minimize(MSE)
    SRCNN_train_set = SRCNN.prepare_training.read_train_set('Set91')
    SRCNN_test_set = SRCNN.prepare_training.read_test_set('baby')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction  = 0.5
    with tf.Session(config = config) as sess:
        with tf.device('/device:GPU:0'):
            sess.run(tf.global_variables_initializer())
            for i in range(1000000):

                bicubiced_LR_batch,SR_batch = SRCNN_train_set.next_batch(100)

                sess.run(optimizer,feed_dict=
                {
                    LR_placeholder : bicubiced_LR_batch,
                    GT_placeholder : SR_batch
                })
                if(i%100==0):
                    bicubiced_LR_Images,GT_SR_Images = SRCNN_test_set.next_batch(1)

                    print('step %d : PSNR: %.5f' % (i, sess.run(PSNR,feed_dict=
                    {
                        LR_placeholder: bicubiced_LR_Images,
                        GT_placeholder: GT_SR_Images
                    })))


if __name__=='__main__':
    main()
