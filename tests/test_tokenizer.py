# coding: utf-8

import unittest

from dnlp.aux2 import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_tokenize(self):
        self.assertTrue(True)

        print(Tokenizer.token('官方太小气，连个膜也不给，wifi经常断89^-^。'))

    def test_c_tokenize(self):
        self.assertTrue(True)

        import re

        EXTRACT_TOKEN_RE = r'([a-z0-9]+)'

        txt = '，wifi经常断a8还不错'

        for x in re.split(EXTRACT_TOKEN_RE, txt):
            print('->', x)


if __name__ == '__main__':
    unittest.main()
