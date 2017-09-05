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

    def test_symbol(self):
        self.addCleanup(True)

        SYMBOL_SOS = '<SOS>'
        SYMBOL_EOS = '<EOS>'

        line = '还不错。'

        line = (SYMBOL_SOS + ' ') * 4 + line + (' ' + SYMBOL_EOS) * 4

        print(line)

    def test_xx(self):
        self.assertTrue(True)

        a = [1, 2, 3, 4, 5]
        print(a[-4:])

    def test_wed_model_segment_pinyin(self):
        self.assertTrue(True)

        from dnlp.wed.model import Model

        Model.segment_pinyin_txt('我更喜欢Iphone，其他的就是8了')

    def test_wed_model_vocab(self):
        self.assertTrue(True)

        from dnlp.wed.model import Model
        model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
        model.build_chars()
        print(model.vocab_chars)

    def test_wed_model_train_data(self):
        self.assertTrue(True)

        from dnlp.wed.model import Model

        model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
        model.build_train_data()

    def test_wed_model_train(self):
        self.assertTrue(True)

        from dnlp.wed.model import Model

        model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
        model.train()
        # model.save()

    def test_predict(self):
        self.assertTrue(True)

        from dnlp.wed.model import Model

        model = Model.load()

        txts = [
            '东西不错',
            '客服服务态度很好',
            '还给赠品了哟。',
            '还行',
            '还不错！',
            '不错。',
            '手机不错。',
            '很漂亮的手机。',
            '手机信号不太好',
            '经常都搜索不到4G。',
            '不错的手机',
            '还没发现有什么问题',
            '下次有机会再购买。',
            '满意的。',
            '机子很好的。',
            '朋友都很喜欢的。',
            '快递员很好的。',
            '第二天就送到了的。',
            '用的不错很好。',
            '好错不明看去。'
        ]

        model.predict(txts)

if __name__ == '__main__':
    unittest.main()
