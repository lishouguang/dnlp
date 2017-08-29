# coding: utf-8

import os
import unittest

from collections import defaultdict

from pypinyin import pinyin
from pypinyin import lazy_pinyin
from pypinyin.utils import simple_seg
from pypinyin.constants import RE_HANS

from dnlp.utils import iter_file
from dnlp.utils import write_file
from dnlp.config import RESOURCE_PATH

from dnlp.nlp.pinyin import tag_pinyin


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_pinyin(self):
        self.assertTrue(True)

        txt = '-中文饕鬄chinese萨比'
        for part in simple_seg(txt):
            if RE_HANS.match(part):
                print(part, '_cn')
            else:
                print(part)

        # print(pinyin(txt))
        # print(lazy_pinyin(txt))

    def test_nlp_pinyin(self):
        self.assertTrue(True)

        txt = '官方太小气，连个膜也不给，wifi经常断。'
        print(tag_pinyin(txt))

    def test_pinyin_create_pinyin(self):
        self.assertTrue(True)

        lines = []

        for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')):
            newparts = []

            for part in simple_seg(line):
                if RE_HANS.match(part):
                    pys = lazy_pinyin(part)
                    newparts.append(' '.join(['{}_{}'.format(z, p) for z, p in zip(part, pys)]))
                else:
                    newparts.append(part)

            lines.append(' '.join(newparts))

        write_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.pinyin.min.txt'), lines)

    def test_pinyin_create_count(self):
        self.assertTrue(True)

        dx = defaultdict(dict)

        for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')):

            for part in simple_seg(line):
                if RE_HANS.match(part):
                    pys = lazy_pinyin(part)
                    for z, p in zip(part, pys):
                        if z not in dx[p]:
                            dx[p][z] = 0

                        dx[p][z] = dx[p][z] + 1

        # for p in dx:
        #     print(p, dx[p].keys())


if __name__ == '__main__':
    unittest.main()
