import cPickle as pkl
import gzip
import os
import sys
import time

import threading
import Queue
import logging

import numpy

import theano
import theano.tensor as T

import tables

logger = logging.getLogger(__name__)

def load_data(path, valid_path=None, test_path=None,
              batch_size=128, n=4, n_words=10000,
              n_gram=True, shortlist_dict=None):
    '''
    Loads the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'

    train = PytablesBitextIterator(batch_size, path, n=n, n_words=n_words, use_infinite_loop=False, shortlist_dict=shortlist_dict)
    valid = PytablesBitextIterator(batch_size, valid_path, n=n, n_words=n_words, use_infinite_loop=False, shortlist_dict=shortlist_dict) if valid_path else None
    test = PytablesBitextIterator(batch_size, test_path, n=n, n_words=n_words, use_infinite_loop=False, shortlist_dict=shortlist_dict) if test_path else None

    return train, valid, test


def get_length(path):
    target_table = tables.open_file(path, 'r')
    target_index = target_table.get_node('/indices')

    return target_index.shape[0]


class PytablesBitextFetcher(threading.Thread):
    def __init__(self, parent, start_offset, max_offset=-1, n=4):
        threading.Thread.__init__(self)
        self.parent = parent
        self.start_offset = start_offset
        self.max_offset = max_offset
        self.n = n

    def run(self):

        diter = self.parent

        driver = None
        if diter.can_fit:
            driver = "H5FD_CORE"

        target_table = tables.open_file(diter.target_file, 'r', driver=driver)
        target_data, target_index = (target_table.get_node(diter.table_name),
            target_table.get_node(diter.index_name))

        data_len = target_index.shape[0]

        offset = self.start_offset
        if offset == -1:
            offset = 0
            self.start_offset = offset
            if diter.shuffle:
                offset = numpy.random.randint(data_len)
        logger.debug("{} entries".format(data_len))
        logger.debug("Starting from the entry {}".format(offset))

        while not diter.exit_flag:
            last_batch = False
            target_ngrams = []

            while len(target_ngrams) < diter.batch_size:
                if offset == data_len or offset == self.max_offset:
                    if diter.use_infinite_loop:
                        offset = self.start_offset
                    else:
                        last_batch = True
                        break

                tlen, tpos = target_index[offset]['length'], target_index[offset]['pos']
                offset += 1

                # discard if there are too many unknown words in the candidate sentence
                cand = numpy.array(target_data[tpos:tpos+tlen])
                if numpy.sum(cand >= diter.n_words - 1) > int(numpy.round(0.1 * tlen)):
                    continue

                target_ngrams.append(target_data[tpos:tpos+tlen])
                # for n-grams
                """
                # for each word, grab n-gram
                for tii in xrange(tlen):
                    if tii < self.n+1:
                        ng = numpy.zeros(self.n+1) # 0 </s>
                        ng[self.n-tii:] = target_data[tpos:tpos+tii+1]
                    else:
                        ng = target_data[tpos+tii-(self.n):tpos+tii+1]
                    if diter.shortlist_dict:
                        count = 0
                        for nn in ng:
                            if nn in diter.shortlist_dict:
                                count += 1
                        if count == 0:
                            continue
                    target_ngrams.append(ng)
                   """

            if len(target_ngrams):
                diter.queue.put([int(offset),target_ngrams])
            if last_batch:
                diter.queue.put([None])
                return


def create_padded_batch_valid(state, x, y, seqlen, return_dict=False):
    """A callback given to the iterator to transform data in suitable format
    :type x: list
    :param x: list of numpy.array's, each array is a batch of phrases
        in some of source languages
    :type y: list
    :param y: same as x but for target languages
    :param new_format: a wrapper to be applied on top of returned value
    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
        OR new_format applied to the tuple
    Notes:
    * actually works only with x[0] and y[0]
    * len(x[0]) thus is just the minibatch size
    * len(x[0][idx]) is the size of sequence idx
    THIS DOESN'T CHECK THE EOS for the validation.
    """

    mx = seqlen
    my = seqlen
    if state['trim_batches']:
        # Similar length for all source sequences
        mx = numpy.minimum(seqlen, max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = numpy.minimum(seqlen, max([len(xx) for xx in y[0]]))+1

    # Batch size
    n = x[0].shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        # if mx is longer than the length of the sequence
        # it wil just the whole sequence ergo :len(x[0][idx])
        if mx < len(x[0][idx]):
            X[:mx, idx] = x[0][idx][:mx]
        else:
            X[:len(x[0][idx]), idx] = x[0][idx][:mx]

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[len(x[0][idx]):, idx] = state['null_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[0][idx]), idx] = 1.
        # Similarly mark the end of phrase
        if len(x[0][idx]) < mx:
            Xmask[len(x[0][idx]), idx] = 1.

    # Fill Y and Ymask in the same way as X and Xmask in the previous loop
    for idx in xrange(len(y[0])):
        Y[:len(y[0][idx]), idx] = y[0][idx][:my]
        if len(y[0][idx]) < my:
            Y[len(y[0][idx]):, idx] = state['null_sym']
        Ymask[:len(y[0][idx]), idx] = 1.
        if len(y[0][idx]) < my:
            Ymask[len(y[0][idx]), idx] = 1.

    null_inputs = numpy.zeros(X.shape[1])
    # Unknown words
    X[X >= state['n_sym']] = state['unk_sym']
    Y[Y >= state['n_sym']] = state['unk_sym']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask


class PytablesBitextIterator_UL(object):

    def __init__(self,
                 batch_size,
                 mode = 'train',
                 state = None,
                 target_file=None,
                 dtype="int64",
                 table_name='/phrases',
                 index_name='/indices',
                 can_fit=False,
                 queue_size=1000,
                 cache_size=1000,
                 val_size=1000,
                 shuffle=True,
                 use_infinite_loop=True,
                 n=4,
                 n_words=-1,
                 shortlist_dict=None):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.exit_flag = False

    def start(self, start_offset=0, max_offset=-1):
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = PytablesBitextFetcher(self, start_offset, max_offset, n=self.n)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):

        batch = self.queue.get()
        if not batch:
            return None
        self.next_offset = batch[0]
        barray = numpy.array(batch[1])
        X = [x[:-1].astype(self.dtype) for x in barray]
        Y = [y[1:].astype(self.dtype) for y in barray]
        assert len(X[0]) == len(Y[0])
        if self.mode == 'valid':
            X = numpy.asarray(X)
            Y = numpy.asarray(Y)
            return create_padded_batch_valid(self.state, [X], [Y],
                                      len(X[0]), return_dict=True)
        else:
            return X, Y
            # get rid of out of vocabulary stuff

