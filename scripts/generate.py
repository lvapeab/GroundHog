#!/usr/bin/python

"""
Script that parses the wikipedia dump, and generates the dataset in a nice
numpy format (i.e. in numpy.npz files).
Call :
    generate_word,py --help
"""
from collections import Counter
import ConfigParser
import optparse
import os
import time
import sys
import numpy
import operator

def construct_vocabulary(dataset, oov_rate, level):
    filename = os.path.join(dataset,  'train')
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' \n ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
    # Order the words
    print ' .. sorting words'
    all_items = Counter(txt).items()
    no_end = [x for x in all_items if x[0] !='\n']
    freqs = [x for x in all_items if x[0] == '\n'] + \
            sorted(no_end,
                   key=lambda t: t[1],
                   reverse=True)
    print ' .. shrinking the vocabulary size'
    # Decide length
    all_freq = float(sum([x[1] for x in freqs]))
    up_to = oov_rate
    '''
    up_to = len(freqs)
    oov = 0.
    remove_word = True
    while remove_word:
        up_to -= 1
        oov += float(freqs[up_to][1])
        if oov / all_freq > oov_rate:
            remove_word = False
    up_to += 1
    '''
    freqs = freqs[:up_to]
    words = [x[0] for x in freqs]
    return dict(zip(words, range(up_to))), [x[1]/all_freq for x in freqs],freqs


def grab_text(path, filename, vocab, oov_default, dtype, level):
    filename = os.path.join(path, filename)
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
        return numpy.asarray(
            [vocab.get(w, oov_default) for w in txt],
            dtype=dtype)
    else:
        return numpy.array(
            [vocab.get(w, oov_default) for w in txt],
            dtype=dtype)


def main(parser):
    o, _ = parser.parse_args()
    dataset = '/data/lisatmp3/firatorh/languageModelling/corpora/wiki'
    print 'Constructing the vocabulary ..'
    vocab, freqs, freq_wd = construct_vocabulary(dataset, o.oov_rate, o.level)
    if o.oov == '-1':
        oov_default = -1
    else:
        oov_default = len(vocab)
    print 'Constructing train set'
    train = grab_text(dataset, 'train', vocab, oov_default, o.dtype, o.level)
    if o.n_chains > 1:
        data_per_chain = train.shape[0] // o.n_chains
        train = train[:data_per_chain * o.n_chains]
        train = train.reshape((data_per_chain, o.n_chains))
        # Do the reshape
    print 'Constructing valid set'
    valid = grab_text(dataset, 'valid', vocab, oov_default, o.dtype, o.level)
    print 'Constructing test set'
    test = grab_text(dataset, 'test', vocab, oov_default, o.dtype, o.level)
    print 'Saving data'
    numpy.savez(o.dest,
                train_words=train,
                valid_words=valid,
                test_words=test,
                oov=oov_default,
                freqs = numpy.array(freqs),
                n_words=len(vocab)+1,
                vocabulary = vocab,
                freq_wd = freq_wd
               )
    dictionary = [y[0] for y in sorted(vocab.items(),key=operator.itemgetter(1))]
    numpy.savez(o.dest+'_vocab', unique_words=dictionary)
    print '... Done'


def get_parser():
    usage = """
This script parses the wikipedia dataset from
http://mattmahoney.net/dc/text.html, and generates more numpy friendly
format of the dataset. Please use this friendly formats as temporary forms
of the dataset (i.e. delete them after you're done).

The script will save the entire file into a numpy .npz file. The file will
contain the following fields:
    'train' : array/matrix where each element (word or letter) is
              represented by an index from 0 to vocabulary size or the
              oov value (out of vocabulary). It is the training data.
    'test' : array where each element (word or letter) is represented by an
             index from 0 to vocabulary size or the oov value. This is the
             test value.
    'test' : array where each element (word or letter) is represented by an
             index from 0 to vocabulary size or the oov value. This is the
             validation set.
    'oov' : The value representing the out of vocabulary word
    'vocab_size' : The size of the vocabulary (this number does not account
                   for oov
    """
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--dest',
                      dest='dest',
                      help=('Where to save the processed dataset (i.e. '
                            'under what name and at what path)'),
                      default='tmp_data')
    parser.add_option('--level',
                      dest='level',
                      help=('Processing level. Either `words` or `letter`. '
                            'If set to word, the result dataset has one '
                            'token per word, otherwise a token per letter'),
                      default='words')
    parser.add_option('--n_chains',
                      dest='n_chains',
                      type="int",
                      help=('Number of parallel chains for the training '
                            'data. The way it works, is that it takes the '
                            'training set and divides it in `n_chains` that '
                            'should be processed in parallel by your model'),
                      default=1)
    # Dataset is already split !
    """
    parser.add_option('--train_size',
                      dest='train_size',
                      type="string",
                      help=('number of samples in the training set. please '
                            'use something like 10m (for 10 millions), 10k '
                            'for (10 thousands). if no letter at the end, '
                            'it is assumed to be an exact number of samples')
                      default='90M')
    parser.add_option('--valid_size',
                      dest='valid_size',
                      type='string',
                      help=('number of samples in the validation set. please '
                            'use something like 10m (for 10 millions), 10k '
                            'for (10 thousands). if no letter at the end, '
                            'it is assumed to be an exact number of samples')
                      default='5M')
    """
    parser.add_option('--oov_index',
                      dest='oov',
                      type='string',
                      help=('index for oov words (in case of word level). '
                            'The accepted values can be `-1`, `last`'),
                      default='-1')
    parser.add_option('--oov_rate',
                      dest='oov_rate',
                      type='int',
                      help=('Defines dictionary size. If for example '
                            'oov_rate is set to 0.01 (meaning 10%) it means '
                            'that we can shrink our dictionary such that '
                            'remaining unrepresented words of the **train** '
                            'set is less then 10%. If set to 0, all words in '
                            'the training set will be added to the '
                            'dictionary'),
                      default=0.)
    parser.add_option('--dtype',
                      dest='dtype',
                      help='dtype in which to store data',
                      default='int32')
    return parser

if __name__ == '__main__':
    main(get_parser())
