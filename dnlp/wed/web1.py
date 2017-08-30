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

import os
import sys
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

from dnlp.config import RESOURCE_PATH
from dnlp.utils import iter_file


SYMBOL_SOS = '<SOS>'
SYMBOL_EOS = '<EOS>'


def build_chars(corpus_file):
    chars = set()

    for line in iter_file(corpus_file):
        chars |= set(line.lower().split())

    # chars.remove('，')
    # chars.remove('。')
    # chars.remove('？')
    # chars.remove('！')

    chars.add(SYMBOL_SOS)
    chars.add(SYMBOL_EOS)

    return sorted(list(chars))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def run():
    corpus_file = os.path.join(RESOURCE_PATH, 'corpus', 'wed', 'std.word.txt')
    chars = build_chars(corpus_file)

    print('total chars:', len(chars))

    char2idx = dict((c, i) for i, c in enumerate(chars))
    idx2char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 5
    step = 2
    histories = []
    next_chars = []

    for line in iter_file(os.path.join(corpus_file)):
        line = (SYMBOL_SOS + ' ') * 4 + line + (' ' + SYMBOL_EOS) * 4
        words = line.split()

        for i in range(0, len(words) - maxlen, step):
            histories.append(words[i: i + maxlen])
            next_chars.append(words[i + maxlen])

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
    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, epochs=1)

        model.save(os.path.join(RESOURCE_PATH, 'model', 'wed', 'wed.epoch_{}.model').format(iteration))

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            history = random.sample(chars, 1)
            generated += history
            print('----- Generating with seed: "' + history + '"')
            sys.stdout.write(generated)

            for i in range(50):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(history):
                    x[0, t, char2idx[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = idx2char[next_index]

                generated += next_char
                history = history[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    run()
