import os
import sys
from six.moves import urllib
import tensorflow as tf
import tarfile

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size for training')
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data', 'the directory of training/test data of cifar10')

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'






























def maybe_download_and_extract():
    dst_dir = FLAGS.data_dir
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    file_name = DATA_URL.split('/')[-1]
    file_path = os.path.join(dst_dir, file_name)
    if not os.path.exists(file_path):
        def _progress(count,block_size,total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (file_name,float(count*block_size/total_size*100)))
            sys.stdout.flush
        downloaded_file_path, _ = urllib.request.urlretrieve(DATA_URL,file_path,_progress)
        print()
        statinfo = os.stat(downloaded_file_path)
        print('Downloaded ', file_name, statinfo.st_size, 'bytes')
    extract_dir_path = os.path.join(dst_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extract_dir_path):
        tarfile.open(file_path,'r:gz').extractall(dst_dir)



def main():
    maybe_download_and_extract()


if __name__ == '__main__':
    main()















