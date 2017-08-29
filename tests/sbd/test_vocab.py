# coding: utf-8

import os
import unittest

from collections import Counter

from matplotlib import pyplot as plt

from dnlp.utils import iter_file
from dnlp.vocab import Vocabulary
from dnlp.config import RESOURCE_PATH


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_chars(self):
        self.assertTrue(True)

        print([_ for _ in u'你好吗？'])

    def test_str(self):
        self.assertTrue(True)

        print('你哈')
        print(type('你好'), len('你好'))
        print(type(u'你好'), len(u'你好'))

    def test_sentence_length(self):
        self.assertTrue(True)

        counter = Counter()

        for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')):
            tokens = Vocabulary.extract_tokens(line)
            counter.update([len(tokens)])

        values = counter.values()
        print('max:', max(values), ' min:', min(values), ' avg:', sum(values)/len(values))

        i = 1
        for t, c in counter.items():
            plt.bar(i, c)
            i += 1

        plt.show()

if __name__ == '__main__':
    unittest.main()
