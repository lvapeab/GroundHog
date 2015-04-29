#!/usr/bin/env python

import argparse
import cPickle
import logging
import time

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    prototype_lm_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)


class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time


class BeamSearch(object):

    def __init__(self, enc_decs):
        self.enc_decs = enc_decs

    def compile(self):
        num_models = len(self.enc_decs)
        self.comp_repr = []
        self.comp_init_states = []
        self.comp_next_probs = []
        self.comp_next_states = []
        for i in xrange(num_models):
            self.comp_repr.append(self.enc_decs[i].create_representation_computer())
            self.comp_init_states.append(self.enc_decs[i].create_initializers())
            self.comp_next_probs.append(self.enc_decs[i].create_next_probs_computer())
            self.comp_next_states.append(self.enc_decs[i].create_next_states_computer())

    def search(self, seq, n_samples, eos_id, unk_id, ignore_unk=False, minlen=1):
        num_models = len(self.enc_decs)
        c = []
        for i in xrange(num_models):
            c.append(self.comp_repr[i](seq)[0])
        states = []
        for i in xrange(num_models):
            states.append(map(lambda x: x[None, :], self.comp_init_states[i](c[i])))
        dim = states[0][0].shape[1]

        # Set initial states of  the language
        # model inside of the decoder
        states_lm = [None] * num_models
        states_mem_lm = [None] * num_models
        for i in xrange(num_models):
            if self.enc_decs[i].state['include_lm']:
                dim_lm = self.enc_decs[i].decoder.state['lm_readout_dim']
                states_lm[i] = numpy.zeros((1, dim_lm), dtype="float32")
                if self.enc_decs[i].state['use_arctic_lm']:
                    states_mem_lm[i] = numpy.zeros((1, dim_lm), dtype="float32")

        num_levels = len(states[0])

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # reset states of the decoders language model
            new_states_lm = [None] * num_models
            new_states_mem_lm = [None] * num_models
            for i in xrange(num_models):
                if self.enc_decs[i].state['include_lm']:
                    dim_lm = self.enc_decs[i].decoder.state['lm_readout_dim']
                    new_states_lm[i] = numpy.zeros((n_samples, dim_lm),dtype="float32")
                    if self.enc_decs[i].state['use_arctic_lm']:
                        new_states_mem_lm[i] = numpy.zeros((n_samples, dim_lm), dtype="float32")

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t: t[-1], trans))
                          if k > 0
                          else numpy.zeros(beam_size, dtype="int64"))

            log_probs = sum(numpy.log(self.comp_next_probs[i](c[i], k, last_words,
                            states_lm[i], states_mem_lm[i], *states[i])[0])
                            for i in xrange(num_models))/num_models

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:, eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = []
            for i in xrange(num_models):
                new_states.append([numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)])
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    for j in xrange(num_models):
                        new_states[j][level][i] = states[j][level][orig_idx]

                for j in xrange(num_models):
                    if self.enc_decs[j].state['include_lm']:
                        new_states_lm[j][i] = states_lm[j][orig_idx]
                        if self.enc_decs[j].state['use_arctic_lm']:
                            new_states_mem_lm[j][i] = states_mem_lm[j][orig_idx]

                inputs[i] = next_word
            for i in xrange(num_models):
                if self.enc_decs[i].state['include_lm']:
                    if self.enc_decs[i].state['use_arctic_lm']:
                        new_states[i], new_states_lm[i], new_states_mem_lm[i] =\
                            self.comp_next_states[i](c[i], k, inputs,
                                                  new_states_lm[i],
                                                  new_states_mem_lm[i],
                                                  *new_states[i])
                    else:
                        new_states[i], new_states_lm[i] =\
                            self.comp_next_states[i](c[i], k,
                                    inputs, new_states_lm[i], new_states_mem_lm[i],
                                    *new_states[i])
                else:
                    new_states[i] = self.comp_next_states[i](c[i], k, inputs,
                                                       new_states_lm[i],
                                                       new_states_mem_lm[i],
                                                       *new_states[i])

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != eos_id:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            for i in xrange(num_models):
                if self.enc_decs[i].state['include_lm']:
                    states[i] = map(lambda x : x[indices], [new_states[i]])
                    states_lm[i] = new_states_lm[i][indices]
                    if self.enc_decs[i].state['use_arctic_lm']:
                        states_mem_lm[i] = new_states_mem_lm[i][indices]
                else:
                    states[i] = map(lambda x : x[indices], new_states[i])

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id, ignore_unk=False, minlen=minlen)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, eos_id=eos_id, unk_id=unk_id, ignore_unk=False, minlen=minlen)
            else:
                logger.warning("No appropriate translation: return empty translation")
                fin_trans = [[]]
                fin_costs = [0.0]

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_trans, fin_costs


def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen


def sample(lm_model, seq, n_samples, eos_id, unk_id,
           sampler=None, beam_search=None,
           ignore_unk=False, normalize=False,
           alpha=1, verbose=False):
    if beam_search:
        sentences = []
        trans, costs = beam_search.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {}".format(costs[i], sentences[i])
        return sentences, costs, trans
    elif sampler:
        raise NotImplementedError
    else:
        raise Exception("I don't know what to do")


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-search) translations from a translation model")
    parser.add_argument("--states", nargs = '+',
            required=True, help="State to use")
    parser.add_argument("--models", nargs = '+', required=True,
            help="path to the models")
    parser.add_argument("--beam-search",
            action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            type=int, help="Beam size")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--n-best", type=int, default=1,
            help="Write n-best list (of size --beam-size)")
    parser.add_argument("--start", type=int, default=0,
            help="For n-best, first sentence id")
    parser.add_argument("--changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()


def main():
    args = parse_args()

    states = []
    for i in xrange(len(args.states)):
        states.append(prototype_state())
        with open(args.states[i]) as src:
            states[i].update(cPickle.load(src))
        states[i].update(eval("dict({})".format(args.changes)))

    logging.basicConfig(
        level=getattr(logging, states[i]['level']),
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    num_models = len(args.models)
    rng = numpy.random.RandomState(states[0]['seed'])
    enc_decs = []
    lm_models = []

    for i in xrange(num_models):
        enc_decs.append(RNNEncoderDecoder(states[i], rng, skip_init=True))
        enc_decs[i].build()
        lm_models.append(enc_decs[i].create_lm_model())
        lm_models[i].load(args.models[i])
    indx_word = cPickle.load(open(states[i]['word_indx'],'rb'))

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_decs)
        beam_search.compile()
    else:
        raise NotImplementedError('use only beam-search')

    idict_src = cPickle.load(open(states[0]['indx_word'],'r'))
    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')
        if args.n_best > 0:
            fidx_scores = open(args.trans + '_SCORES', 'w')
        start_time = time.time()

        # TODO: this may not be true
        eos_id = states[0]['null_sym_target']
        unk_id = states[0]['unk_sym_target']

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            if states[0]['source_encoding'] == 'utf8':
                seqin = line.strip().decode('utf-8')
            else:
                seqin = line.strip()

            seq, parsed_in = parse_input(states[0], indx_word, seqin, idx2word=idict_src)
            if args.verbose:
                print "Parsed Input:", parsed_in
            trans, costs, _ = sample(lm_models[0], seq, n_samples, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk,
                    normalize=args.normalize, verbose=args.verbose,
                    eos_id=eos_id, unk_id=unk_id)
            nbest_idx = numpy.argsort(costs)[:args.n_best]
            for j, best in enumerate(nbest_idx):
                if states[0]['target_encoding'] == 'utf8':
                    txt2write = trans[best].encode('utf8').replace(" ", "")
                else:
                    txt2write = trans[best]
                print >>ftrans, txt2write

                if args.n_best > 0:
                    print >>fidx_scores, "%d|||%s|||%f" % (i, txt2write, costs[best])

                if args.verbose:
                    print "Translation:", txt2write

                total_cost += costs[best]

            if (i + 1) % 100 == 0:
                ftrans.flush()
                logger.debug("Current speed is {} per sentence".
                        format((time.time() - start_time) / (i + 1)))

            if args.n_best > 0:
                fidx_scores.flush()
            ftrans.flush()

        print "Total cost of the translations: {}".format(total_cost)

        if args.n_best > 0:
            fidx_scores.close()
        fsrc.close()
        ftrans.close()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
