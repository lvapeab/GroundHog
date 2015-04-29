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

import lstmlm as arcLM

from experiments.nmt.train import BleuValidator
from experiments.nmt.sample import \
        BeamSearch as BeamSearchBare
from experiments.nmt.sampleShallowArcticLM import \
        BeamSearch as BeamSearchShallowLM

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
    parser.add_argument("--n-best",
        type=int, default=1,
        help="Output n-best lists")
    parser.add_argument("--score",
        action="store_true", default=False,
        help="Write scores to auxiliary file")
    parser.add_argument("changes",  nargs="*",
        help="Changes to state", default="")
    # shallow fusion with auxiliary lm parameters
    parser.add_argument("--lm-state",
        default=None,
        help="State to use as an auxiliary LM")
    parser.add_argument("--lm-model",
        default=None,
        help="Model to use as an auxiliary LM")
    parser.add_argument("--eta",
        default=0.5, type=float,
        help="Balancing parameter between TM and LM log-probs")
    parser.add_argument("--append-bleu",
        action="store_true", default=False,
        help="append bleu score to output file")
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
        #state.update(eval("dict({})".format(change)))
        state.update(eval("{%s}" % change))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))
    rng = numpy.random.RandomState(state['seed'])

    state['use_noise'] = False
    state['weight_noise'] = False
    state['weight_noise_rec'] = False

    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    logging.debug("Loading TM model from %s" % args.model)
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

    # check if we are doing shallow fusion
    beam_search = []
    if args.lm_state is not None and args.lm_model is not None:
        logging.debug("Create auxiliary language model from arctic")
        # load language model parameters
        aux_lm = {}
        logging.debug("Loading LM model from %s" % args.lm_model)
        aux_lm['model_options'] = arcLM.load_options(args.lm_model)
        aux_lm['model_tparams'] = arcLM.init_tparams(arcLM.load_params(args.lm_model,
                                           arcLM.init_params(aux_lm['model_options'])))
        beam_search = BeamSearchShallowLM(enc_dec, aux_lm, args.eta, True) #args.score)
    else: # bare TM model
        beam_search = BeamSearchBare(enc_dec)

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
                                           normalize=args.normalize,
                                           score=args.score,
                                           n_best=args.n_best)
        val_bleu_validator.best_bleu = numpy.inf
        val_bleu_validator()
        val_bleu_score = val_bleu_validator.val_bleu_curve[-1]
        if args.append_bleu:
            os.rename(args.val_out,
                      '%s-%sBLEU%2.2f' % (args.val_out,
                                          'NORMALIZED_' if args.normalize else '',
                                          val_bleu_score))

    if tst_src is not None and tst_gld is not None and tst_src != '-1':
        state['validation_set_out'] = args.tst_out
        state['validation_set'] = tst_src
        state['validation_set_grndtruth'] = tst_gld
        tst_bleu_validator = BleuValidator(state,
                                           lm_model,
                                           beam_search,
                                           verbose=True,
                                           ignore_unk=args.ignore_unk,
                                           normalize=args.normalize,
                                           score=args.score,
                                           n_best=args.n_best)
        tst_bleu_validator.best_bleu = numpy.inf
        tst_bleu_validator()
        tst_bleu_score = tst_bleu_validator.val_bleu_curve[-1]
        if args.append_bleu:
            os.rename(args.tst_out,
                      '%s-%sBLEU%2.2f' % (args.tst_out,
                                          'NORMALIZED_' if args.normalize else '',
                                          tst_bleu_score))

    print 'val bleu = %2.2f' % val_bleu_score
    print 'tst bleu = %2.2f' % tst_bleu_score


if __name__ == "__main__":
    main()

