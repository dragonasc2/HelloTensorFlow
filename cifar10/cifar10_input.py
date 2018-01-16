import os
import tensorflow as tf

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


class CIFAR10Record:
    height = 32
    width = 32
    depth = 3


def read_cifar10(filename_queue):
    result = CIFAR10Record();
    result.height = 32
    result.width = 32
    result.depth = 3
    label_bytes = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value,tf.uint8)
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]),tf.int32
    )

    depth_major = tf.reshape(
        tf.stride_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), [result.depth, result.height, result.width]
    )
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3*batch_size
        )
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def distored_inputs(data_dir, batch_size):
    file_names = [os.path.join(data_dir,'data_batch_%' % i) for i in range(1,6)]
    for f in file_names:
        if not tf.gfile.Exists(f):
            raise ValueError('filed to find file '+ f )
    filename_queue = tf.train.string_input_producer(file_names)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distored_image = tf.random_crop(reshaped_image,[height, width, 3])

    distored_image = tf.image.random_brightness(distored_image, max_delta=63)
    distored_image = tf.image.random_contrast(distored_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distored_image)

    float_image.set_shape([height,width,3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)



def main():
    pass

if __name__ == '__main__':
    pass







