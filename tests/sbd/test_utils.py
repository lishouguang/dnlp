# coding: utf-8

import os
import unittest

from dnlp.config import RESOURCE_PATH
from dnlp.utils import save_obj, read_obj


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_save_read_obj(self):
        self.assertTrue(True)

        a = [1, 2, 3, 5]
        print(a)

        ff = os.path.join(RESOURCE_PATH, 'tmp', 'a.list')

        save_obj(a, ff)

        a = read_obj(ff)
        print(a)

if __name__ == '__main__':
    unittest.main()
