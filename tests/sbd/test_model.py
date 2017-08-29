# coding: utf-8

import os
import unittest

from dnlp.config import RESOURCE_PATH
from dnlp.sbd.model import Model


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_model_train_dataset(self):
        self.assertTrue(True)

        model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
        model.create_standard_dataset()
        model.create_train_dataset()

    def test_train(self):
        self.assertTrue(True)

        model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
        model.train()
        model.save()

    def test_ctable(self):
        self.assertTrue(True)

        model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
        model._build_ctable()
        print(model.ctable.chars)

    def test_model(self):
        pass


if __name__ == '__main__':
    unittest.main()
