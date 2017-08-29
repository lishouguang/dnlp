# coding: utf-8

import logging

logging.basicConfig(level=logging.INFO)

import os
import difflib
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Activation

from dnlp.utils import iter_file
from dnlp.utils import write_file
from dnlp.utils import save_obj
from dnlp.utils import read_obj
from dnlp.sbd import preprocessing
from dnlp.config import RESOURCE_PATH

from dnlp.aux2 import Tokenizer
from dnlp.aux2 import CharacterTable


logger = logging.getLogger(__file__)


class Model(object):

    def __init__(self, raw_file, max_sentence_length=100, hidden_size=128, batch_size=128, layers=3):
        self._raw_file = raw_file

        self._workspace = os.path.join(RESOURCE_PATH, 'model')
        self._clean_file = os.path.join(self._workspace, 'data.clean')
        self._label_file = os.path.join(self._workspace, 'data.label')
        self._model_file = os.path.join(self._workspace, 'sbd.model')
        self._keras_model_file = os.path.join(self._workspace, 'keras.model')

        self._max_sentence_length = max_sentence_length

        self._ctable = CharacterTable()
        self._model = Sequential()

        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._layers = layers

    def create_standard_dataset(self):
        """
        读取原始的文本，清洗后，提取出规范句子，存储到文件
        :return:
        """
        sentences = []

        for line in iter_file(self._raw_file):
            txt = preprocessing.clean_txt(line)
            sents = preprocessing.extract_standard_sentences(txt)
            sentences += [sent for sent in sents if preprocessing.is_meaningful(sent)]

        write_file(self._clean_file, sentences)

    def create_train_dataset(self):
        """
        标注训练数据，分字->标注
        <E>表示句子的最后一个字，<M>表示句子的非最后一个字

        特别好，发货很快，赞。 => <M> <M> <E> <M> <M> <M> <E> <E> 。
        :return:
        """
        lines = []

        for line in iter_file(self._clean_file):
            result = Labeler.label(line)
            token = ' '.join([t for t, _ in result])
            sequence = ' '.join([seq for _, seq in result])
            lines.append('%s\t%s' % (token, sequence))

        write_file(self._label_file, lines)

    def train(self):

        logger.info('build char table...')
        self._build_ctable()

        logger.info('create standard dataset...')
        self.create_standard_dataset()

        logger.info('create train dataset...')
        self.create_train_dataset()

        logger.info('create train vector...')
        X_train, y_train = self._build_train_data()

        logger.info('build and train model...')
        self._train_keras_model(X_train, y_train)

    def predict_sequence(self, txt):

        ctable = self._ctable

        MAX_SENTENCE_LENGTH = self._max_sentence_length
        VOCAB_SIZE = len(ctable)

        X = np.zeros((1, MAX_SENTENCE_LENGTH, VOCAB_SIZE), dtype=np.int8)

        logger.info('Vectorization...')
        tokens = Tokenizer.token(txt)
        X[0] = ctable.encode(tokens, MAX_SENTENCE_LENGTH)

        y_pred = self._model.predict(X, verbose=1)
        logger.info(y_pred)

        labels = []
        for xx in y_pred[0]:
            labels.append(ctable.indices_char[np.argmax(xx)])

        return labels

    def predict_txt(self, txt):
        labels = self.predict_sequence(txt)
        tokens = Tokenizer.token(txt)
        return ''.join([t if l == Labeler.SYMBOL_MIDDLE else '{}，'.format(t) for t, l in zip(tokens, labels[:len(tokens)])])

    def save(self):
        save_obj(self, self._model_file)
        self._model.save(self._keras_model_file)

    @classmethod
    def load(cls, keras_model_file=None):
        """
        :rtype: Model
        """
        model = Model(None)

        logger.info('loading model...')
        model = read_obj(model._model_file)

        assert isinstance(model, Model)

        if keras_model_file is None:
            keras_model_file = model._keras_model_file

        logger.info('loading keras model...')
        model._model = keras.models.load_model(keras_model_file)
        return model

    def _build_ctable(self):
        self._ctable = CharacterTable.build(self._clean_file, special_chars=[Labeler.SYMBOL_END, Labeler.SYMBOL_MIDDLE])

    def _build_train_data(self):
        VOCAB_SIZE = len(self._ctable)
        MAX_SENTENCE_LENGTH = self._max_sentence_length

        SAMPLE_NUMBER = 0
        for line in iter_file(self._label_file):
            if line:
                SAMPLE_NUMBER += 1

        logger.info('Vectorization...')
        X = np.zeros((SAMPLE_NUMBER, MAX_SENTENCE_LENGTH, VOCAB_SIZE), dtype=np.int8)
        y = np.zeros((SAMPLE_NUMBER, MAX_SENTENCE_LENGTH, VOCAB_SIZE), dtype=np.int8)

        for i, line in enumerate(iter_file(self._label_file)):
            if not line:
                continue

            sentence, sequence = line.split('\t')

            X[i] = self._ctable.encode(sentence.split(), MAX_SENTENCE_LENGTH)
            y[i] = self._ctable.encode(sequence.split(), MAX_SENTENCE_LENGTH)

        logger.info('Shuffle...')
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Explicitly set apart 10% for validation data that we never train over.
        # split_at = len(X) - len(X) // 10
        # (X_train, x_val) = X[:split_at], X[split_at:]
        # (y_train, y_val) = y[:split_at], y[split_at:]
        X_train, y_train = X, y

        return X_train, y_train

    def _train_keras_model(self, X_train, y_train):
        HIDDEN_SIZE = self._hidden_size
        BATCH_SIZE = self._batch_size
        LAYERS = self._layers
        VOCAB_SIZE = len(self._ctable)

        logger.info('Build model...')

        self._model = Sequential()
        model = self._model

        # encoder RNN
        model.add(LSTM(HIDDEN_SIZE, input_shape=(None, VOCAB_SIZE), return_sequences=True))
        # model.add(RNN(HIDDEN_SIZE, input_shape=(MAX_SENTENCE_LENGTH, VOCAB_SIZE)))
        # decoder

        for _ in range(LAYERS - 1):
            model.add(LSTM(HIDDEN_SIZE, return_sequences=True))

        model.add(TimeDistributed(Dense(VOCAB_SIZE)))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.summary()

        # train
        epoch = 0
        while True:
            logger.info('epoch: {} \n\n'.format(epoch))
            model.fit(X_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=1)

            epoch += 1

            if epoch % 10 == 0:
                model_file = os.path.join(self._workspace, 'sbd.epoch_{}.model'.format(epoch))
                model.save(model_file)


class Labeler(object):

    SYMBOL_END = '<E>'
    SYMBOL_MIDDLE = '<M>'

    CODES = {
        SYMBOL_MIDDLE: [1],
        SYMBOL_END: [2]
    }

    @classmethod
    def label(cls, txt):

        tokens = Tokenizer.token(txt)
        token_txt = ''.join(tokens)

        differ = difflib.Differ()
        indices = [i for i, d in enumerate(differ.compare(token_txt, txt)) if len(d.strip()) == 3 and d.strip().startswith('+')]

        labels = []
        l = 0
        for token in tokens:
            l += len(token)

            if l in indices:
                labels.append(Labeler.SYMBOL_END)
                l += 1
            else:
                labels.append(Labeler.SYMBOL_MIDDLE)

        labels[-1] = Labeler.SYMBOL_END
        return [tp for tp in zip(tokens, labels)]

    @classmethod
    def encode_1(cls, sequences, max_len):

        if len(sequences) > max_len:
            sequences = sequences[:max_len]

        real = np.array([Labeler.CODES.get(s) for s in sequences])

        pad_len = max_len - len(sequences)
        if pad_len > 0:
            pad = np.zeros((pad_len, 1))
            return np.concatenate((real, pad))
        else:
            return real


def run_test():
    # model = Model(os.path.join(RESOURCE_PATH, 'corpus', 'std.min.txt'))
    # model._build_ctable()
    # model.train()
    # model.save()

    txt = '昨天刚买的今天就坏了质量太差了'

    model = Model.load(keras_model_file=os.path.join(RESOURCE_PATH, 'model', 'sbd.epoch_80.model'))
    sequence = model.predict_sequence(txt)
    print(sequence)

    ptxt = model.predict_txt(txt)
    print(ptxt)

if __name__ == '__main__':
    run_test()
