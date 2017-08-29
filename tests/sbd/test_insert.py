# coding: utf-8

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_array_insert(self):
        self.assertTrue(True)

        a = [1, 2, 3]
        print(a)

        a.insert(0, 0)
        print(a)

if __name__ == '__main__':
    unittest.main()
