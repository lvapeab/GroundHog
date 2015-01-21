#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import pprint
import os

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state

from experiments.nmt.train import BleuValidator
from experiments.nmt.sample import BeamSearch

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--proto",  default="prototype_search_state",
        help="Prototype state to use for state")
    parser.add_argument("--val-src", default=None,
        help="validation set source file")
    parser.add_argument("--val-gld", default=None,
        help="validation set gold file")
    parser.add_argument("--val-out", default='./DEV_TRANSLATION',
        help="validation translation output")
    parser.add_argument("--tst-src", default=None,
        help="test set source file")
    parser.add_argument("--tst-gld", default=None,
        help="test set gold file")
    parser.add_argument("--tst-out", default='./TST_TRANSLATION',
        help="test translation output")
    parser.add_argument("--beam-size", default=None,
            type=int, help="Beam size")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    # this loads the state specified in the prototype
    state = getattr(experiments.nmt, args.proto)()
    # this is based on the suggestion in the README.md in this foloder
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model)

    val_src = None
    if state.has_key('validation_set') and state['validation_set'] is not None:
        val_src = state['validation_set']
    if args.val_src is not None:
        val_src = args.val_src

    val_gld = None
    if state.has_key('validation_set_grndtruth') and state['validation_set_grndtruth'] is not None:
        val_gld = state['validation_set_grndtruth']
    if args.val_gld is not None:
        val_gld = args.val_gld

    tst_src = None
    if state.has_key('test_set') and state['test_set'] is not None:
        tst_src = state['test_set']
    if args.tst_src is not None:
        tst_src = args.tst_src

    tst_gld = None
    if state.has_key('test_set_grndtruth') and state['test_set_grndtruth'] is not None:
        tst_gld = state['test_set_grndtruth']
    if args.tst_gld is not None:
        tst_gld = args.tst_gld

    if args.beam_size is not None:
        state['beam_size'] = args.beam_size

    beam_search = BeamSearch(enc_dec)
    beam_search.compile()
    val_bleu_score = 0.0
    tst_bleu_score = 0.0
    if val_src is not None and val_gld is not None and val_src != '-1':
        state['validation_set_out'] = args.val_out
        state['validation_set'] = val_src
        state['validation_set_grndtruth'] = val_gld
        val_bleu_validator = BleuValidator(state,
                                           lm_model,
                                           beam_search,
                                           verbose=True,
                                           ignore_unk=args.ignore_unk,
                                           normalize=args.normalize)
        val_bleu_validator.best_bleu = numpy.inf
        val_bleu_validator()
        val_bleu_score = val_bleu_validator.val_bleu_curve[-1]
        os.rename(args.val_out,'%s-BLEU%2.2f' % (args.val_out,val_bleu_score))

    if tst_src is not None and tst_gld is not None and tst_src != '-1':
        state['validation_set_out'] = args.tst_out
        state['validation_set'] = tst_src
        state['validation_set_grndtruth'] = tst_gld
        tst_bleu_validator = BleuValidator(state,
                                           lm_model,
                                           beam_search,
                                           verbose=True,
                                           ignore_unk=args.ignore_unk,
                                           normalize=args.normalize)
        tst_bleu_validator.best_bleu = numpy.inf
        tst_bleu_validator()
        tst_bleu_score = tst_bleu_validator.val_bleu_curve[-1]
        os.rename(args.tst_out,'%s-BLEU%2.2f' % (args.tst_out,tst_bleu_score))

    print 'val bleu = %2.2f' % val_bleu_score
    print 'tst bleu = %2.2f' % tst_bleu_score


if __name__ == "__main__":
    main()

