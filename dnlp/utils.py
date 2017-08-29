# coding: utf-8

import codecs
import pickle


def iter_file(_f):
    with open(_f, 'rb') as ff:
        for line in ff:
            yield line.strip().decode('utf-8')


def write_file(_f, lines):
    with codecs.open(_f, 'w', encoding='utf-8') as ff:
        for line in lines:
            ff.write('%s\n' % line)


def save_obj(obj, sfile):
    with open(sfile, 'wb') as sf:
        pickle.dump(obj, sf)


def read_obj(sfile):
    with open(sfile, 'rb') as sf:
        return pickle.load(sf)
