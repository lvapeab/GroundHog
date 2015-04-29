#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import numpy

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_phrase_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def parse_args():
    parser = argparse.ArgumentParser(
            "Replace UNK by original word")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--mapping",
            help="Top1 unigram mapping (Source to target)")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File of translated sentences")
    parser.add_argument("--new-trans",
            help="File to save new translations in")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--heuristic", type=int, default=0,
            help="0: copy, 1: Use dict, 2: Use dict only if lowercase \
            Used only if a mapping is given. Default is 0.")
    parser.add_argument("--topn-file",
         type=str,
         help="Binarized topn list for each source word (Vocabularies must correspond)")
    parser.add_argument("model_path",
            help="Path to the model")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()


def main():
    args = parse_args()

    state = prototype_phrase_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    if 'save_algo' not in state:
        state['save_algo'] = 0
    if 'save_gs' not in state:
        state['save_gs'] = 0
    if 'save_iter' not in state:
        state['save_iter'] = -1

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)

    if args.mapping:
        with open(args.mapping, 'rb') as f:
            mapping = cPickle.load(f)
        heuristic = args.heuristic
    else:
        heuristic = 0

    word2indx_src = cPickle.load(open(state['word_indx'], 'rb'))
    word2indx_trg = cPickle.load(open(state['word_indx_trgt'], 'rb'))

    idict_src = cPickle.load(open(state['indx_word'],'r'))
    idict_trg = cPickle.load(open(state['indx_word_target'],'r'))

    if state['oov'] in word2indx_trg: # 'UNK' may be in the vocabulary
        word2indx_trg[state['oov']] = 1

    unk_id = state['unk_sym_target']

    compute_probs = enc_dec.create_probs_computer(return_alignment=True)

    if args.source and args.trans and args.new_trans:

        with open(args.source, 'r') as src_file:
            with open(args.trans, 'r') as trans_file:
                with open(args.new_trans, 'w') as new_trans_file:
                    while True:
                        src_line = src_file.readline()
                        trans_line = trans_file.readline()
                        if src_line == '' or trans_line == '':
                            break
                        src_seq, src_words = parse_input(state, word2indx_src, src_line.strip())
                        src_words.append('<eos>')
                        trans_seq, trans_words = parse_input(state, word2indx_trg, trans_line.strip())
                        trans_words.append('<eos>')
                        probs, alignment = compute_probs(src_seq, trans_seq)
                        #alignment = alignment[:,:,0]
                        alignment = alignment[:,:-1,0] # Remove source <eos>
                        hard_alignment = numpy.argmax(alignment, 1)
                        new_trans_words = []
                        for i in xrange(len(trans_words) - 1): # -1 : Don't write <eos>
                            if trans_seq[i] == unk_id:
                                UNK_src = src_words[hard_alignment[i]]
                                if heuristic == 0:
                                    new_trans_words.append(UNK_src)
                                elif heuristic == 1:
                                    # Use the most likely translation (with t-table).
                                    # If not found, copy the source word.
                                    if UNK_src in mapping:
                                        new_trans_words.append(mapping[UNK_src])
                                    else:
                                        new_trans_words.append(UNK_src)
                                elif heuristic == 2:
                                    # Use t-table if the source word starts with a lowercase letter.
                                    # Otherwise copy
                                    if UNK_src in mapping and UNK_src.decode('utf-8')[0].islower():
                                        new_trans_words.append(mapping[UNK_src])
                                    else:
                                        new_trans_words.append(UNK_src)
                            else:
                                new_trans_words.append(trans_words[i])
                        to_write = ''
                        for i, word in enumerate(new_trans_words):
                            to_write = to_write + word
                            if i < len(new_trans_words) - 1:
                                to_write += ' '
                        to_write += '\n'
                        new_trans_file.write(to_write)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
