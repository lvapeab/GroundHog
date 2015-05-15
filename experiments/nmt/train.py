#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import re
import numpy
import time
import experiments

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.trainer.SGD_rmspropv2 import SGD as SGD_rmsprop
from groundhog.trainer.SGD_adam import SGD as SGD_adam

from groundhog.mainLoop import MainLoop
from experiments.nmt import\
        RNNEncoderDecoder, prototype_state, get_batch_iterator,\
        sample, sampleShallowArcticLM, parse_input

from experiments.nmt.sample import BeamSearch

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx: self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx: self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                self.__print_samples("Input", x_words, self.state['source_encoding'])
                self.__print_samples("Target", y_words, self.state['target_encoding'])
                self.model.get_samples(
                    self.state['seqlen'] + 1,
                    self.state['n_samples'], x[:len(x_words)],
                    self.model.cross_dict)

                sample_idx += 1

    def __print_samples(self, output_name, words, encoding):
        if encoding == "utf8":
            print u"{}: {}".format(output_name, " ".join(words))
        elif encoding == "ascii":
            print "{}: {}".format(output_name, " ".join(words))
        else:
            print "Unknown encoding {}".format(encoding)


class BleuValidator(object):
    """
    Object that evaluates the bleu score on the validation set.
    Opens the subprocess to run the validation script, and
    keeps track of the bleu scores over time
    """
    def __init__(self, state, lm_model,
                 beam_search, ignore_unk=False,
                 normalize=False, verbose=False,
                 score=False, n_best=1):
        """
        Handles normal book-keeping of state variables,
        but also reloads the bleu scores if they exists

        :param state:
            a state in the usual groundhog sense
        :param lm_model:
            a groundhog language model
        :param beam_search:
            beamsearch object used for sampling
        :param ignore_unk
            whether or not to ignore unknown characters
        :param normalize
            whether or not to normalize the score by the length
            of the sentence
        :param verbose
            whether or not to also write the ranslation to the file
            specified by validation_set_out
        :param score
            whether or not to also write the scores and corresponding
            line indices to a separate file
        :param b_best
            n-best list outputs to a separate file
        """

        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

        self.indx_word = cPickle.load(open(state['word_indx'],'rb'))
        self.idict_src = cPickle.load(open(state['indx_word'],'r'))
        self.n_samples = state['beam_size']
        self.best_bleu = 0

        self.val_bleu_curve = []
        if state['reload']:
            try:
                bleu_score = numpy.load(state['prefix'] + 'val_bleu_scores.npz')
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()
                self.best_bleu = self.validation_set_grndtruth.max()
                logger.debug("BleuScores Reloaded")
            except:
                logger.debug("BleuScores not Found")

        if state['char_based_bleu']:
            self.multibleu_cmd = ['perl', state['bleu_script'], '-char', state['validation_set_grndtruth'], '<']
        else:
            self.multibleu_cmd = ['perl', state['bleu_script'], state['validation_set_grndtruth'], '<']

        if verbose:
            if 'validation_set_out' not in state.keys():
                self.state['validation_set_out'] = state['prefix'] + 'validation_out.txt'

        # This is a condition for shallow fusion
        if isinstance(self.beam_search, sampleShallowArcticLM.BeamSearch):
            self.sample_fn = sampleShallowArcticLM.sample
        else:
            self.sample_fn = sample.sample


    def __call__(self):
        """
        Opens the file for the validation set and creates a subprocess
        for the multi-bleu script.

        Returns a boolean indicating whether the current model should
        be saved.
        """

        print "Started Validation: "
        val_start_time = time.time()
        fsrc = open(self.state['validation_set'], 'r')
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.state['validation_set_out'], 'w')
        if self.score:
            fscores = open(self.state['validation_set_out'] + '_SCORES', 'w')

        for i, line in enumerate(fsrc):
            """
            Load the sentence, retrieve the sample, write to file
            """

            if self.state['source_encoding'] == 'utf8':
                seqin = line.strip().decode('utf-8')
            else:
                seqin = line.strip()

            seq, parsed_in = parse_input(self.state, self.indx_word, seqin, idx2word=self.idict_src)

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs, _ = self.sample_fn(
                self.lm_model, seq,
                self.n_samples,
                beam_search=self.beam_search,
                ignore_unk=self.ignore_unk,
                normalize=self.normalize,
                cross_dict=self.lm_model.cross_dict
                if hasattr(self.lm_model, 'cross_dict') else None)

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]
                except ValueError:
                    print "Could not fine a translation for line: {}".format(i+1)
                    trans_out = u'UNK' if self.state['target_encoding'] == 'utf8' else 'UNK'

                if j == 0:
                    # Write to subprocess and file if it exists
                    if self.state['target_encoding'] == 'utf8' and \
                        self.state['char_based_bleu']:
                        print >> mb_subprocess.stdin, trans_out.encode('utf8').replace(" ","")

                        if self.verbose:
                            print  >> ftrans, trans_out.encode('utf8').replace(" ","")
                    elif self.state['target_words_segmented']:
                        print >> mb_subprocess.stdin, \
                                self.append_suffixes(trans_out)#.encode(self.state['target_encoding']))
                        if self.verbose:
                            print >> ftrans, \
                                    self.append_suffixes(trans_out)#.encode(self.state['target_encoding']))
                    else:
                        print >> mb_subprocess.stdin, trans_out
                        if self.verbose:
                            print >> ftrans, trans_out

                if self.score:
                    if self.state['target_encoding'] == 'utf8' and \
                        self.state['char_based_bleu']:
                        txt2write = trans_out.encode('utf8').replace(" ","")
                    elif self.state['target_words_segmented']:
                        txt2write =  self.append_suffixes(trans_out)#.encode(self.state['target_encoding']))
                    else:
                        txt2write = trans_out

                    print >> fscores, "%d|||%s|||%f" % (i, txt2write, costs[best])

            if i != 0 and i % 100 == 0:
                print "Translated {} lines of validation set...".format(i)

            mb_subprocess.stdin.flush()

            #if i == 200:
            #    break

        print "Total cost of the validation: {}".format(total_cost)
        if self.score:
            fscores.close()
        fsrc.close()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        print "output ", stdout
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        print "Validation Took: {} minutes".format(float(time.time() - val_start_time) / 60.)
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        print bleu_score
        mb_subprocess.terminate()
        # Determine whether or not we should save
        if self.best_bleu < bleu_score:
            self.best_bleu = bleu_score
            return True
        return False

    def append_suffixes(self,trans):
        '''
        Suffix merger for segmented words, looks for suffixes in <tag:_sfx_>
        pattern and appends _sfx_ to its corresponding stem

        :param trans:
            sentence to be merged, string

        '''
        out = []
        for word in trans.split():
            if word.startswith('<') and word.endswith('>'):
                sfx = re.search(r':(.*)\>',word)
                if sfx is not None:
                    out.append(sfx.group(1))
            else: # it is a stem
                if not out:
                    out.append(' ')
                out.append(word)
        return ''.join(out)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_search_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
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
    enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()

    # If we are going to use validation with the bleu script, we
    # will need early stopping
    bleu_validator = None
    if state['bleu_script'] is not None and state['validation_set'] is not None\
        and state['validation_set_grndtruth'] is not None:

        logger.debug("Building beam search enc_dec")

        # copy dictionary
        state_validation = state.copy()
        state_validation['weight_noise'] = False
        state_validation['weight_noise_rec'] = False
        state_validation['weight_noise_amount'] = 0.
        state_validation['use_noise'] = False

        # make beam search with a separate enc_dec
        enc_dec_validation = RNNEncoderDecoder(state_validation, rng, skip_init=True)
        enc_dec_validation.build(use_noise=False)

        '''
        lm_model_validation = enc_dec_validation.create_lm_model()
        enc_dec_val_lm_params = enc_dec_validation.lm_model.params
        enc_dec_validation.params = enc_dec.params
        enc_dec_val_params = enc_dec_validation.predictions.params
        nparams = len(enc_dec_validation.predictions.params)
        for i in xrange(nparams):
            enc_dec_val_params[i] = enc_dec.predictions.params[i]
            enc_dec_val_lm_params[i] = enc_dec.lm_model.params[i]
        import ipdb;ipdb.set_trace()
        '''

        beam_search = BeamSearch(enc_dec_validation)
        beam_search.compile()
        bleu_validator = BleuValidator(enc_dec_validation.state,
                                       lm_model,
                                       beam_search,
                                       normalize=enc_dec_validation.state['normalized_bleu'] \
                                                    if 'normalized_bleu' in \
                                                    enc_dec_validation.state \
                                                    else True,
                                       verbose=enc_dec_validation.state['output_validation_set'])

    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")

    algo = eval(state['algo'])(lm_model, state, train_data)
    logger.debug("Run training")

    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            bleu_val_fn = bleu_validator,
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0 #and state['validation_set'] is not None
                else None)

    if state['reload'] or state['reload_lm']:
        exclude_params=[]
        if enc_dec.state['random_readout'] and\
                enc_dec.state['train_only_readout']:
            exclude_params = set(lm_model.params)-set(enc_dec.lm_model.exclude_params)
        if enc_dec.state['random_readout'] and\
                not enc_dec.state['train_only_readout']:
            exclude_params = set(enc_dec.decoder.readout_params)
        main.load(exclude_params=exclude_params)

    if state['loopIters'] > 0:
        main.main()

if __name__ == "__main__":
    main()
