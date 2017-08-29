# coding: utf-8

import os

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding

from dnlp.utils import iter_file
from dnlp.config import RESOURCE_PATH

from dnlp.aux2 import CharacterTable


def run():
    '''
    参考https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    训练Language Model，预测句子/拼音概率
    :return:
    '''

    ctable = CharacterTable.build(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
    print(ctable.chars)

    vocab_size = len(ctable)
    hidden_size = 100

    model = Sequential()
    model.add(Embedding(vocab_size, hidden_size))

    pass


if __name__ == '__main__':
    run()
