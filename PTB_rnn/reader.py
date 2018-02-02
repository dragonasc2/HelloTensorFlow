# Author : Dragon_n

import collections
import os
import sys
from six.moves import urllib
from datetime import datetime
import tarfile

import tensorflow as tf


def maybe_download_and_extract(dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_download_file_name = os.path.join(dst_dir, 'PTB simple-examples.tgz')
    if not os.path.exists(dst_download_file_name):
        t_start = datetime.now()
        def _report_progress(block_num, block_size, total_size):
            sys.stdout.write('\r>> downloading %s, %.2f%%, cost:%s' % (dst_download_file_name, block_num*block_size/total_size*100, datetime.now()-t_start))
            sys.stdout.flush()
        dst_download_file_name, _ = urllib.request.urlretrieve('http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz', dst_download_file_name, _report_progress)
        print()
        tarobj = tarfile.open(dst_download_file_name, 'r:gz')
        for tarinfo in tarobj:
            tarobj.extract(tarinfo.name, dst_dir)
        tarobj.close()


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace('\n', '<eos>').split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    counter_pairs = sorted(counter.items(), key=lambda x : (-x[1], x[0]))

    words, _ = list(zip(*counter_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word]          for word in data if word in word_to_id]


def ptb_raw_data(download_path):
    """
    Args:
        download_path : string path for PTB data to download and extract

    Returns:
        tuple(train_data, valid_data, test_data, word_to_id)



    """
    maybe_download_and_extract(download_path)
    data_path = os.path.join(download_path, 'simple-examples\data')
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id

def ptb_producer(raw_data, batch_size, num_steps):
    with tf.name_scope("PTBProducer", values=[raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0:batch_len * batch_size], [batch_size, batch_len])
        # -1 is because y is right shifted x by 1
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name='epoch_size')

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])

        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y



if __name__ == '__main__':
    download_path = 'PTB_data'
    print(ptb_raw_data(download_path))

pass