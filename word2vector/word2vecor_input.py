import sys
import os
import tensorflow as tf
import numpy as np
import zipfile
from six.moves import urllib

DATA_URL = 'http://mattmahoney.net/dc/'
VOCABULARY_SIZE = 50000


class Word2vector_input:
    def __init__(self):
        file_path = self._maybe_download('word2vector_data', 'text8.zip')
        words = self._read_data(file_path)
        self._data, self._word_count_list, self._word_index_map, self._index_word_map = self._build_dataset(words, VOCABULARY_SIZE)
        self._data_ptr = 0

        pass

    def next_batch(self, batch_size, LR_window):
        assert batch_size % (2*LR_window) == 0
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * LR_window + 1 # [LR_window target LR_window]
        if self._data_ptr < LR_window:
            self._data_ptr = LR_window
        elif self._data_ptr + LR_window >= len(self._data):
            self._data_ptr = LR_window
        ptr = 0
        for i in range(batch_size // (2*LR_window)):
            window = self._data[self._data_ptr:self._data_ptr + span]
            context_index = [w for w in range(span) if w != LR_window]
            for j in context_index:
                batch[ptr] = window[LR_window]
                labels[ptr, 0] = window[j]
                ptr += 1
            self._data_ptr += 1
            if self._data_ptr + LR_window >= len(self._data):
                self._data_ptr = LR_window
        return batch, labels

    @property
    def num_data(self):
        return len(self._data)

    def map_index_to_word(self, index):
        return self._index_word_map[index]

    def map_word_to_index(self, word):
        return self._word_index_map[word]

    @staticmethod
    def _maybe_download(dst_dir, dst_file_name):
        expected_bytes = 31344016
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        dst_path = os.path.join(dst_dir, dst_file_name)
        if not os.path.exists(dst_path):
            def _report_progress(block_num, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.2f%%' % (DATA_URL + dst_file_name, block_num * block_size / total_size * 100))
                sys.stdout.flush()
            print()
            dst_path, _ = urllib.request.urlretrieve(DATA_URL + dst_file_name, dst_path, _report_progress)
        statinfo = os.stat(dst_path)
        if(statinfo.st_size != expected_bytes):
            raise Exception('received %d bytes, not match to expected %d bytes' % (statinfo.st_size, expected_bytes))
        return dst_path

    @staticmethod
    def _read_data(file_path):
        with zipfile.ZipFile(file_path) as f:
            return str(f.read(f.namelist()[0])).split()

    @staticmethod
    def _build_dataset(words, num_words):
        '''
        transform the str words to int words.
        :param words: list of str
        :param num_words: number of words to extract. Only the most common number of words will be extract
        :return: data: transformed words (in int format);
        extracted_word_count_list: [word, count] list ;
        extracted_word_index_map:  word -> index map (dict of int->str)
        index_word_map: index->word map (list of str)
        '''
        word_counter_map = dict()
        total_count = 0
        for word in words:
            total_count += 1
            if word not in word_counter_map:
                word_counter_map[word] = 0
            word_counter_map[word] += 1
        word_count_list = list()
        for (w, c) in word_counter_map.items():
            word_count_list.append((w, c))
        word_count_list.sort(key=lambda word_count_pair : word_count_pair[1], reverse=True)
        extracted_word_index_map = dict()

        extract_count = 0
        extracted_word_count_list = [['UNK', -1]]
        index_word_map = ['UNK']
        data = list()
        for i in range(min(num_words, len(word_count_list))-1):
            word_count_pair = word_count_list[i]
            extracted_word_index_map[word_count_pair[0]] = i+1
            index_word_map.append(word_count_pair[0])
            extract_count += word_count_pair[1]
            extracted_word_count_list.append(word_count_pair)
        extracted_word_count_list[0][1] = total_count - extract_count

        for word in words:
            index = extracted_word_index_map.get(word, 0)
            data.append(index)
        return data, extracted_word_count_list, extracted_word_index_map, index_word_map



if __name__ == '__main__':
    input = Word2vector_input()
    batch, labels = input.next_batch(16, 2)
    pass