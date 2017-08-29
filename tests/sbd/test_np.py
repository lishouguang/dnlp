# coding: utf-8

import unittest

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_np(self):
        self.assertTrue(True)

        X0 = np.empty((0, 3))
        print(X0)

        X1 = np.random.randint(100, size=(5, 3))

        print('------------')
        print(np.concatenate((X0, X1)))

        X2 = np.random.randint(100, size=(5, 3))
        print(X1)
        print(X2)
        X3 = np.concatenate((X1, X2))
        print(X3)

if __name__ == '__main__':
    unittest.main()
