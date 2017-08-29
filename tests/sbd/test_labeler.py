# coding: utf-8

import unittest

from dnlp.aux2 import Labeler


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_encode(self):
        self.assertTrue(True)

        sequences = 'M M M M M E M M E M M E'.split()
        print(Labeler.encode(sequences, 40))

if __name__ == '__main__':
    unittest.main()
