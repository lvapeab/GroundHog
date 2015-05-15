#!/usr/bin/env python

'''
Generates a joint vocabulary pickle file given input text file.
If the given vocabulary size is not covered by the joint dictionary, then the most frequent
not-added words are appended from either of input files until the joint vocabulary size is met
Usage:
    generate_joint_vocab.py -v 30000 train.tags.zh-en.en.tok train.tags.tr-en.en.tok
        -p : pickle input files using joint vocabulary
        -o : overwrite earlier created files
        -a <aux_file> : pickle given auxiliary file

Adapted from GroundHog/experiments/nmt/preprocess/preprocess.py
'''

import argparse
import cPickle
import gzip
import bz2
import logging
import os
import operator

from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType('r'), nargs="*",
                        help="The input files")
    parser.add_argument("-d", "--dictionary", default='joint_vocab.pkl',
                        help="the name of the pickled binarized text file")
    parser.add_argument("-v", "--vocab", type=int, metavar="N",
                        help="limit vocabulary size to this number, which must "
                          "include BOS/EOS and OOV markers")
    parser.add_argument("-l", "--limit-before", type=int, metavar="N",
                        default=0,
                        help="limit vocabulary sizei before taking the union to this number, which must "
                          "include BOS/EOS and OOV markers")
    parser.add_argument("-p", "--pickle", action="store_true",
                        help="pickle input text files as a list of lists of ints")
    parser.add_argument("-t", "--char", action="store_true",
                        help="character-level processing")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="overwrite earlier created files, also forces the "
                             "program not to reuse count files")
    parser.add_argument("-a", "--auxiliary-text", default=None,
                        help="auxiliary file to be binarized according to the "
                         "original dictionary generated from input files")
    parser.add_argument("-e", "--external-vocab", default=None,
                        help="external vocabulary to binarize input text files")
    return parser.parse_args()

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
        logger.warning("Not saving %s, already exists" % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s" % filename)
        else:
            logger.info("Saving to %s" % filename)
        with open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def create_dictionary():
    # Part I: Counting the words
    counters = []
    sentence_counts = []
    for input_file, base_filename in zip(args.input, base_filenames):
        count_filename = base_filename + '.count.pkl'
        input_filename = os.path.basename(input_file.name)
        if os.path.isfile(count_filename) and not args.overwrite:
            logger.info("Loading word counts for %s from %s"
                        % (input_filename, count_filename))
            with open(count_filename, 'rb') as f:
                counter = cPickle.load(f)
            sentence_count = sum([1 for line in input_file])
        else:
            logger.info("Counting words in %s" % input_filename)
            counter = Counter()
            sentence_count = 0
            for line in input_file:
                words = None
                if args.char:
                    words = list(line.strip().decode('utf-8'))
                else:
                    words = line.strip().split(' ')
                counter.update(words)
                sentence_count += 1
        counters.append(counter)
        sentence_counts.append(sentence_count)
        logger.info("%d unique words in %d sentences with a total of %d words."
                    % (len(counter), sentence_count, sum(counter.values())))
        input_file.seek(0)

    # Part II: Combining the counts by taking intersection
    logger.info('Taking intersection of all vocabularies')
    if args.limit_before>0:
        for ii in xrange(len(counters)):
            counters[ii] = Counter(
                        dict(counters[ii].most_common(args.limit_before)))

    joint_dict = reduce(operator.or_,counters)

    # Part III: Creating the dictionary
    if args.vocab is not None:
        if args.vocab <= 2:
            logger.info('Building a dictionary with all joint unique words')
            args.vocab = len(joint_dict) + 2
        vocab_count = joint_dict.most_common(args.vocab - 2)
        if len(vocab_count) < args.vocab - 2:
            curr_words   = [i[0] for i in vocab_count]
            surplus_dict = reduce(operator.or_,counters).most_common()
            surplus_dict = [(word,count) for word,count in surplus_dict
                                if word not in curr_words ]
            vocab_count.append(surplus_dict[:(args.vocab - 2 - len(vocab_count))])

        logger.info("Creating dictionary of %s most common words, covering :" % (args.vocab))
        for ctr, base_filename in zip(counters, base_filenames):
            logger.info("  %2.2f%% of the file %s" %
                       (100.0 * (1 - (sum((ctr - joint_dict).values())*1.0 / sum(ctr.values()))),
                       base_filename))
    else:
        logger.info("Creating dictionary of all words")
        vocab_count = joint_dict.most_common()
    vocab = {'<UNK>': 1, '<S>': 0, '</S>': 0}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 2
    safe_pickle(vocab, args.dictionary)
    return joint_dict, sentence_counts, counters, vocab

def binarize():
    for input_file, base_filename, sentence_count in \
            zip(args.input, base_filenames, sentence_counts):
        input_filename = os.path.basename(input_file.name)
        logger.info("Binarizing %s" % (input_filename))
        binarized_corpus = []
        ngram_count = 0
        for sentence_count, sentence in enumerate(input_file):
            if args.char:
                words = list(sentence.strip().decode('utf-8'))
            else:
                words = sentence.strip().split(' ')
            binarized_sentence = [vocab.get(word, 1) for word in words]
            binarized_corpus.append(binarized_sentence)
        # Output
        safe_pickle(binarized_corpus, base_filename + '.pkl')
        input_file.seek(0)

def binarize_aux():

    base_filename = os.path.basename(args.auxiliary_text)
    logger.info("Binarizing %s." % (base_filename))
    binarized_corpus = []
    unk_count = 0
    fin  = open(args.auxiliary_text,'r')

    # read line
    while 1:
        sentence = fin.readline()
        if not sentence:
            break
        words = sentence.strip().split(' ')
        binarized_sentence = [vocab.get(word, 1) for word in words]
        binarized_corpus.append(binarized_sentence)
        unk_count += binarized_sentence.count(1)

    fin.close()
    # endfor sentence in input_file
    logger.info("#Unknown words in auxilary-file %d." % (unk_count))
    # Output
    safe_pickle(binarized_corpus, base_filename + '.pkl')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('generate_joint_dict')
    args = parse_args()
    base_filenames = open_files()
    logger.info("Creating dictionary")
    if args.external_vocab:
        vocab = cPickle.load( open(args.external_vocab, "r" ) )
    else:
        combined_counter, sentence_counts, counters, vocab = create_dictionary()
    if args.pickle:
        binarize()
    if args.auxiliary_text:
        binarize_aux()
    logger.info("Done")
