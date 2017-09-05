# coding: utf-8

from __future__ import print_function

import re
import os
import sys
import math
import random
import copy
import numpy as np

from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

from dnlp.config import RESOURCE_PATH
from dnlp.utils import iter_file
from dnlp.utils import write_file
from dnlp.utils import save_obj
from dnlp.utils import read_obj
from dnlp.nlp.pinyin import tag_pinyin


class Model(object):

    SYMBOL_SOS = '<SOS>'
    SYMBOL_EOS = '<EOS>'

    SYMBOL_UNKNOW = '<UNK>'

    SYMBOL_ENV = '<ENV>'
    SYMBOL_NUM = '<NUM>'
    SYMBOL_ENV_NUM = '<ENV_NUM>'

    SYMBOL_PUN = '<PUN>'

    def __init__(self, corpus_file, vocab_file=None, maxlen=5, step=1):
        if not vocab_file:
            vocab_file = corpus_file

        self._maxlen = maxlen
        self._step = step

        self._chars = []
        self._char2idx = dict()
        self._idx2char = dict()

        self._model = Sequential()

        self._corpus_file = corpus_file
        self._vocab_file = vocab_file

        self._workspace = os.path.join(RESOURCE_PATH, 'model', 'wed')

        self._char_file = os.path.join(self._workspace, 'wed.chars')
        self._char2idx_file = os.path.join(self._workspace, 'wed.char2idx')
        self._idx2char_file = os.path.join(self._workspace, 'wed.idx2char')

        self._model_file = os.path.join(self._workspace, 'wed.model')
        self._keras_model_file = os.path.join(self._workspace, 'wed.kmodel')

        pass

    @property
    def vocab_chars(self):
        return self._chars

    def vocab_char2idx(self, char):
        return self._char2idx.get(char, self._char2idx[Model.SYMBOL_UNKNOW])

    def vocab_idx2char(self, idx):
        return self._idx2char[idx]

    @classmethod
    def segment_pinyin_txt(cls, txt):
        tps = tag_pinyin(txt.lower())
        words = []
        pinyins = []
        for tp in tps:
            if tp[1] is not None:
                words.append(tp[0])
                pinyins.append(tp[1])
            else:
                for x in re.split(r'([a-z0-9]+)', tp[0]):
                    if x:
                        # words.append(x)
                        # pinyins.append(x)

                        symbol = Model.SYMBOL_ENV_NUM
                        if re.match(r'^[a-z]+$', x):
                            symbol = Model.SYMBOL_ENV
                        elif re.match(r'^[0-9]+$', x):
                            symbol = Model.SYMBOL_NUM
                        elif re.match(r'^[a-z0-9]$', x):
                            symbol = Model.SYMBOL_ENV_NUM
                        elif re.match(r'^[。？！!?，,]$', x):
                            symbol = Model.SYMBOL_PUN

                        words.append(symbol)
                        pinyins.append(symbol)

        return words, pinyins

    def build_chars(self):
        chars = set()

        for line in iter_file(self._vocab_file):
            words, pinyins = Model.segment_pinyin_txt(line)
            chars |= set([w for w in words])

        # chars.remove('，')
        # chars.remove('。')
        # chars.remove('？')
        # chars.remove(a'！')

        chars.add(Model.SYMBOL_SOS)
        chars.add(Model.SYMBOL_EOS)
        chars.add(Model.SYMBOL_UNKNOW)
        chars.add(Model.SYMBOL_ENV)
        chars.add(Model.SYMBOL_NUM)
        chars.add(Model.SYMBOL_ENV_NUM)
        chars.add(Model.SYMBOL_PUN)

        chars = sorted(list(chars))
        char2idx = dict((c, i) for i, c in enumerate(chars))
        idx2char = dict((i, c) for i, c in enumerate(chars))

        self._chars = chars
        self._char2idx = char2idx
        self._idx2char = idx2char

    def build_history_nextchars(self, words):
        histories = []
        next_chars = []

        pad_num = self._maxlen - 1
        words = [Model.SYMBOL_SOS] * pad_num + words + [Model.SYMBOL_EOS] * pad_num

        for i in range(0, len(words) - self._maxlen, self._step):
            histories.append(words[i: i + self._maxlen])
            next_chars.append(words[i + self._maxlen])

        return histories, next_chars

    def build_train_data(self):

        histories = []
        next_chars = []

        # 对语料文本进行分字、标注拼音
        for line in iter_file(self._corpus_file):
            words, pinyins = self.segment_pinyin_txt(line)

            histories_, next_chars_ = self.build_history_nextchars(words)
            histories += histories_
            next_chars += next_chars_

        print('history sequences:', len(histories))

        print('Vectorization...')

        X = np.zeros((len(histories), self._maxlen, len(self._chars)), dtype=np.bool)
        y = np.zeros((len(histories), len(self._chars)), dtype=np.bool)

        for i, history in enumerate(histories):
            for t, char in enumerate(history):
                X[i, t, self.vocab_char2idx(char)] = 1
            y[i, self.vocab_char2idx(next_chars[i])] = 1

        return X, y

    def train(self):

        # 构建词表
        self.build_chars()

        print('Build model...')

        model = self._model
        model.add(LSTM(128, input_shape=(self._maxlen, len(self._chars))))
        model.add(Dense(len(self._chars)))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        def sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        X, y = self.build_train_data()
        for iteration in range(1, 1000):
            print()
            print('-' * 50)
            print('Iteration', iteration)

            X, y = shuffle(X, y)

            model.fit(X, y, batch_size=128, epochs=1)

            model.save(os.path.join(self._workspace, 'wed.epoch_{}.model').format(iteration))

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print()
                print('----- diversity:', diversity)

                generated = []
                # history = [SYMBOL_SOS, SYMBOL_SOS, SYMBOL_SOS] + random.sample(chars, 1)
                history = [Model.SYMBOL_SOS] * (self._maxlen - 1)
                generated += history
                print('----- Generating with seed: "' + ' '.join(history) + '"')
                sys.stdout.write(' '.join(generated))

                for i in range(50):
                    x = np.zeros((1, self._maxlen, len(self._chars)))
                    for t, char in enumerate(history):
                        x[0, t, self.vocab_char2idx(char)] = 1.

                    preds = model.predict(x, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = self._idx2char[next_index]

                    generated += [next_char]
                    history = history[-3:] + [next_char]

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()

    def predict(self, txts):

        scores = []
        for txt in txts:

            if not txt:
                scores.append(None)
                continue

            # TODO 文本预处理，清洗/添加标点，根据句号分句

            words, pinyins = self.segment_pinyin_txt(txt)

            histories, next_chars = self.build_history_nextchars(words)

            # print numpy时显示全部数据
            # np.set_printoptions(threshold=np.nan)

            probs = []
            for history, next_char in zip(histories, next_chars):

                x = np.zeros((1, self._maxlen, len(self._chars)), dtype=np.bool)
                for i, word in enumerate(history):
                    x[0, i, self.vocab_char2idx(word)] = 1

                pred = self._model.predict(x)[0]

                prob = pred[self.vocab_char2idx(next_char)]
                probs.append(prob)

                if prob == 0:
                    print('prob is zero!')
                    print(history, next_char)

                # print(history, next_char, prob, score)

                # for idx in np.argsort(pred)[-1:-10:-1]:
                #     print(idx2char[idx], pred[idx])

            # print(txt, 'score:', score)
            if 0 in probs:
                scores.append(None)
            else:
                scores.append(sum(math.log(p) for p in probs) / len(probs))

        return scores

    def save(self):

        self._model.save(self._keras_model_file)
        self._model = None

        save_obj(self, self._model_file)

        # 保存词表信息
        # save_obj(self._chars, self._char_file)
        # save_obj(self._char2idx, self._char2idx_file)
        # save_obj(self._idx2char, self._idx2char_file)

    @classmethod
    def load(cls, model_file=os.path.join(RESOURCE_PATH, 'model', 'wed', 'wed.model')
             , keras_model_file=os.path.join(RESOURCE_PATH, 'model', 'wed', 'wed.kmodel')):
        """
        :param model_file: Model对象保存的文件
        :param keras_model_file: keras model保存的文件
        :return:
        :rtype: Model
        """
        kmodel = keras.models.load_model(keras_model_file)

        model = read_obj(model_file)
        model._model = kmodel

        return model
