# coding: utf-8

import re
import os
import unittest

from dnlp import sbd1
from dnlp.config import RESOURCE_PATH


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_ctable_init(self):
        self.assertTrue(True)

        chars = '0123456789+ '
        ctable = sbd1.CharacterTable(chars)
        print(ctable.chars)

    def test_build_ctable(self):
        self.assertTrue(True)

        corpus_file = os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')
        ctable = sbd1.CharacterTable.build(corpus_file)
        print(ctable.chars)


if __name__ == '__main__':
    unittest.main()
