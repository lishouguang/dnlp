# coding: utf-8

'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function

import re
import os
import sys
import math
import random
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
from dnlp.nlp.pinyin import tag_pinyin


SYMBOL_SOS = '<SOS>'
SYMBOL_EOS = '<EOS>'


def create_train_data():
    word_lines = []
    pinyin_lines = []

    for line in iter_file(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt')):
        words, pinyins = create_txt_data(line)

        word_lines.append(words)
        pinyin_lines.append(pinyins)

    # write_file(os.path.join(RESOURCE_PATH, 'corpus', 'wed', 'std.word.txt'), word_lines)
    # write_file(os.path.join(RESOURCE_PATH, 'corpus', 'wed', 'std.pinyin.txt'), pinyin_lines)
    return word_lines, pinyin_lines


def create_txt_data(line):
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
    return words, pinyins


def build_chars(corpus):
    chars = set()

    for words in corpus:
        chars |= set([w.lower() for w in words])

    # chars.remove('，')
    # chars.remove('。')
    # chars.remove('？')
    # chars.remove('！')

    chars.add(SYMBOL_SOS)
    chars.add(SYMBOL_EOS)

    chars = sorted(list(chars))
    char2idx = dict((c, i) for i, c in enumerate(chars))
    idx2char = dict((i, c) for i, c in enumerate(chars))

    return chars, char2idx, idx2char


def create_history_word(words, history_len, maxlen, step):
    histories = []
    next_chars = []

    pad_num = maxlen - 1
    words = [SYMBOL_SOS] * pad_num + words + [SYMBOL_EOS] * pad_num

    for i in range(0, len(words) - history_len, step):
        histories.append(words[i: i + history_len])
        next_chars.append(words[i + history_len])

    return histories, next_chars


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train():
    corpus_words, corpus_pinyins = create_train_data()
    chars, char2idx, idx2char = build_chars(corpus_words)

    print('total chars:', len(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 5
    step = 1
    histories = []
    next_chars = []

    for words in corpus_words:
        histories_, next_chars_ = create_history_word(words, maxlen, maxlen, step)
        histories += histories_
        next_chars += next_chars_

    print('history sequences:', len(histories))

    print('Vectorization...')

    X = np.zeros((len(histories), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(histories), len(chars)), dtype=np.bool)

    for i, history in enumerate(histories):
        for t, char in enumerate(history):
            X[i, t, char2idx[char]] = 1
        y[i, char2idx[next_chars[i]]] = 1

    print('Build model...')

    # build the model: a single LSTM
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # train the model, output generated text after each iteration
    for iteration in range(1, 1000):
        print()
        print('-' * 50)
        print('Iteration', iteration)

        X, y = shuffle(X, y)

        model.fit(X, y, batch_size=128, epochs=1)

        model.save(os.path.join(RESOURCE_PATH, 'model', 'wed', 'wed.epoch_{}.model').format(iteration))

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = []
            # history = [SYMBOL_SOS, SYMBOL_SOS, SYMBOL_SOS] + random.sample(chars, 1)
            history = [SYMBOL_SOS, SYMBOL_SOS, SYMBOL_SOS, SYMBOL_SOS]
            generated += history
            print('----- Generating with seed: "' + ' '.join(history) + '"')
            sys.stdout.write(' '.join(generated))

            for i in range(50):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(history):
                    x[0, t, char2idx[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = idx2char[next_index]

                generated += [next_char]
                history = history[-3:] + [next_char]

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


def predict(txts, model, chars, char2idx, idx2char):
    maxlen = 5
    step = 2

    for txt in txts:

        # TODO 文本预处理，清洗/添加标点，根据句号分句

        words, pinyins = create_txt_data(txt)

        histories, next_chars = create_history_word(words, maxlen, step)

        # print numpy时显示全部数据
        np.set_printoptions(threshold=np.nan)

        score = 0
        for history, next_char in zip(histories, next_chars):

            x = np.zeros((1, maxlen, len(chars)), dtype=np.bool)
            for i, word in enumerate(history):
                x[0, i, char2idx[word]] = 1

            pred = model.predict(x)[0]

            prob = pred[char2idx[next_char]]
            score += math.log(prob)
            # print(history, next_char, prob, score)

            # for idx in np.argsort(pred)[-1:-10:-1]:
            #     print(idx2char[idx], pred[idx])

        print(txt, 'score:', score)


def predict_x():
    corpus_words, corpus_pinyins = create_train_data()
    chars, char2idx, idx2char = build_chars(corpus_words)

    model = keras.models.load_model(os.path.join(RESOURCE_PATH, 'model', 'wed', 'wed.epoch_368.model'))

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

    predict(txts, model, chars, char2idx, idx2char)


if __name__ == '__main__':
    # train()
    predict_x()
