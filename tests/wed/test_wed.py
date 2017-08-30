# coding: utf-8

import os
import re
import unittest

from dnlp.utils import iter_file
from dnlp.utils import write_file
from dnlp.config import RESOURCE_PATH

from dnlp.nlp.pinyin import tag_pinyin


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_create_data(self):
        self.assertTrue(True)

        word_lines = []
        pinyin_lines = []

        for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')):
            tps = tag_pinyin(line.lower())

            words = []
            pinyins = []

            for tp in tps:
                if tp[1] is not None:
                    words.append(tp[0])
                    pinyins.append(tp[1])
                else:
                    for x in re.split(r'([a-z0-9]+)', tp[0]):
                        if x:
                            words.append(x)
                            pinyins.append(x)

            word_lines.append(' '.join(words))
            pinyin_lines.append(' '.join(pinyins))

        write_file(os.path.join(RESOURCE_PATH, 'corpus', 'wed', 'std.word.txt'), word_lines)
        write_file(os.path.join(RESOURCE_PATH, 'corpus', 'wed', 'std.pinyin.txt'), pinyin_lines)

if __name__ == '__main__':
    unittest.main()
