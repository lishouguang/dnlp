# coding: utf-8

import re
import os
import unittest

from dnlp.utils import iter_file
from dnlp.utils import write_file
from dnlp.utils import save_obj
from dnlp.utils import read_obj
from dnlp.config import RESOURCE_PATH


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_tokenizer(self):
        self.assertTrue(True)

        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences

        def iter_txt(text_file):
            RE = r'([\u4e00-\u9fa5])|([a-z0-9]+)'

            with open(text_file, 'rb') as sf:
                for line in sf:
                    line = line.strip().decode('utf-8')
                    yield ' '.join([t for tp in re.findall(RE, line) for t in tp if t])

        tokenizer = Tokenizer(num_words=2000)
        tokenizer.fit_on_texts(iter_txt(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')))
        sequences = tokenizer.texts_to_sequences(['这 个 手 机 真 不 xxoo 错', '这 个 手 机 真 不 错'])
        print(sequences)
        sequences = pad_sequences(sequences, maxlen=10)
        print(sequences)

    def test_create_train_data(self):
        self.assertTrue(True)

        from dnlp.aux2 import Labeler

        lines = []
        for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')):
            result = Labeler.label(line)
            token = ' '.join([t for t, _ in result])
            sequence = ' '.join([seq for _, seq in result])
            lines.append('%s\t%s' % (token, sequence))

        write_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.label.txt'), lines)

    def test_create_train_data2(self):
        self.assertTrue(True)

        s = []
        y = []
        for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.label.txt')):
            sequence_x, sequence_y = line.split('\t')
            s.append(sequence_x)
            y.append(sequence_y)

        save_obj(s, os.path.join(RESOURCE_PATH, 'corpus', 'sequence.s'))
        save_obj(y, os.path.join(RESOURCE_PATH, 'corpus', 'sequence.y'))

    def test_read_train_data(self):
        self.assertTrue(True)

        s = read_obj(os.path.join(RESOURCE_PATH, 'corpus', 'sequence.s'))
        y = read_obj(os.path.join(RESOURCE_PATH, 'corpus', 'sequence.y'))

        print(len(s))
        print(s[0])
        print(len(y))
        print(y[0])


if __name__ == '__main__':
    unittest.main()
