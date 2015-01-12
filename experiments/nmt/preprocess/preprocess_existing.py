#!/usr/bin/env python

import argparse
import cPickle
import gzip
import bz2
import logging
import os
import string

import numpy
import tables
import pdb

from collections import Counter
from operator import add
from numpy.lib.stride_tricks import as_strided

parser = argparse.ArgumentParser(
    description="""
This takes a list of .txt or .txt.gz files and does word counting and
creating a dictionary (potentially limited by size). It uses this
dictionary to binarize the text into a numeric format (replacing OOV
words with 1) and create n-grams of a fixed size (padding the sentence
with 0 for EOS and BOS markers as necessary). The n-gram data can be
split up in a training and validation set.

The n-grams are saved to HDF5 format whereas the dictionary, word counts
and binarized text are all pickled Python objects.
""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("input", type=argparse.FileType('r'), nargs="+",
                    help="The input files")
parser.add_argument("-b", "--binarized-text", default='binarized_text.pkl',
                    help="the name of the pickled binarized text file")
parser.add_argument("-d", "--vocab", default='vocab.pkl',
                    help="the location of the pickled vocab file")
parser.add_argument("-t", "--char", default=False,
                    help="character-level processing")

def open_files():
    base_filenames = []
    for i, input_file in enumerate(args.input):
        dirname, filename = os.path.split(input_file.name)
        if filename.split(os.extsep)[-1] == 'gz':
            base_filename = filename.rstrip('.gz')
        elif filename.split(os.extsep)[-1] == 'bz2':
            base_filename = filename.rstrip('.bz2')
        else:
            base_filename = filename
        if base_filename.split(os.extsep)[-1] == 'txt':
            base_filename = base_filename.rstrip('.txt')
        if filename.split(os.extsep)[-1] == 'gz':
            args.input[i] = gzip.GzipFile(input_file.name, input_file.mode,
                                          9, input_file)
        elif filename.split(os.extsep)[-1] == 'bz2':
            args.input[i] = bz2.BZ2File(input_file.name, input_file.mode)
        base_filenames.append(base_filename)
    return base_filenames


def safe_pickle(obj, filename):
    if os.path.isfile(filename) and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s." % filename)
        else:
            logger.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def safe_hdf(array, name):
    if os.path.isfile(name + '.hdf') and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (name + '.hdf'))
    else:
        if os.path.isfile(name + '.hdf'):
            logger.info("Overwriting %s." % (name + '.hdf'))
        else:
            logger.info("Saving to %s." % (name + '.hdf'))
        with tables.openFile(name + '.hdf', 'w') as f:
            atom = tables.Atom.from_dtype(array.dtype)
            filters = tables.Filters(complib='blosc', complevel=5)
            ds = f.createCArray(f.root, name.replace('.', ''), atom,
                                array.shape, filters=filters)
            ds[:] = array

def process_utf8(line):
    return [c for c in line.strip().decode('utf8', "replace")
                            if (c not in string.ascii_letters) and c != ' ']

def binarize_external(input_filenames, vocab_location, output_name, split_type):
    """
    Helper function
    """
    vocab = cPickle.load(open(vocab_location, 'r'))
    binarized_corpus = []
    ctr = 0
    for input_file in input_filenames:
        print "processing : {}".format(input_file.name)
        for sentence in input_file:
            if split_type:
                words = list(sentence.strip().decode('utf-8'))
            else:
                words = sentence.strip().split(' ')
            binarized_sentence = [vocab.get(word, 1) for word in words]
            binarized_corpus.append(binarized_sentence)

    safe_pickle(binarized_corpus, output_name)

if __name__ == "__main__":
    #globals
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('preprocess')
    args = parser.parse_args()
    base_filenames = open_files()
    binarize_external(args.input, args.vocab, args.binarized_text, args.char)
