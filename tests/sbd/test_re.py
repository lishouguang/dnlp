# coding: utf-8

import re
import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_char(self):
        self.assertTrue(True)

        s = '你好吗Iphone8 plus，什么时候上市'.lower()
        for x in re.findall(r'([\u4e00-\u9fa5])|([a-z0-9]+)', s):
            for y in x:
                print(y)


if __name__ == '__main__':
    unittest.main()
