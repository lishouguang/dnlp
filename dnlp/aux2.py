# coding: utf-8

import re
import numpy as np

from dnlp.utils import iter_file
from dnlp.utils import write_file


class Tokenizer(object):

    EXTRACT_TOKEN_RE = r'([\u4e00-\u9fa5])|([a-z0-9]+)'

    @classmethod
    def token(cls, txt):
        return [t for tp in re.findall(Tokenizer.EXTRACT_TOKEN_RE, txt) for t in tp if t]


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    SYMBOL_UNK = '<UNK>'

    def __init__(self, chars=''):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    @classmethod
    def build(cls, corpus_file, special_chars=[]):
        ctable = CharacterTable()

        chars = set()
        for line in iter_file(corpus_file):
            chars |= set(Tokenizer.token(line))

        chars.add(' ')

        if special_chars:
            for sc in special_chars:
                chars.add(sc)

        # TODO UNK WORD
        chars.add(CharacterTable.SYMBOL_UNK)

        ctable.chars = sorted(chars)

        ctable.char_indices = dict((c, i) for i, c in enumerate(ctable.chars))
        ctable.indices_char = dict((i, c) for i, c in enumerate(ctable.chars))

        return ctable

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        # TODO UNK WORD
        unk_index = self.char_indices.get(CharacterTable.SYMBOL_UNK)

        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C[:num_rows]):
            # TODO UNK WORD
            x[i, self.char_indices.get(c, unk_index)] = 1
            # x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

    def __len__(self):
        return len(self.chars)
