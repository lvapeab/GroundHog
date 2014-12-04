#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    prototype_lm_state,\
    parse_input,\
    LM_builder

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

    def __init__(self, enc_dec, aux_lm=None, eta=None, score=False):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']
        self.aux_lm = aux_lm
        self.eta    = eta
        self.score = score

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

        # compile sampling and prob functions if necessary
        if self.aux_lm:
            self.comp_next_probs_lm = self.aux_lm.create_next_probs_computer()
            self.comp_next_states_lm = self.aux_lm.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1):
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        # Set initial states of  the language
        # model inside of the decoder
        states_lm = None
        if self.enc_dec.state['include_lm']:
            dim_lm = self.enc_dec.decoder.state_lm['dim']
            states_lm = numpy.zeros((1,dim_lm),dtype="float32")

        # Set initial states of the auxiliary
        # language mode, independent of decoder
        states_aux_lm = None
        if self.aux_lm:
            dim_aux_lm = self.aux_lm.state['dim']
            states_aux_lm = numpy.zeros((1,dim_aux_lm),dtype="float32")
        if self.score:
            score_lm = [[]]
            score_tm = [[]]
            fin_score_lm = []
            fin_score_tm = []

        num_levels = len(states)

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # reset states of the decoders language model
            new_states_lm = None
            if self.enc_dec.state['include_lm']:
                new_states_lm = numpy.zeros((n_samples, dim_lm), dtype="float32")

            # reset states of the auxiliary language model
            new_states_aux_lm = None
            if self.aux_lm:
                new_states_aux_lm = numpy.zeros((n_samples, dim_aux_lm), dtype="float32")

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))

            log_probs_tm = numpy.log(self.comp_next_probs(c, k, last_words, states_lm, *states)[0])

            # get log probability given last words and previous hidden states
            # and then fuse it with TM log probability by their geometric mean
            if self.aux_lm:
                log_probs_lm = numpy.log(self.comp_next_probs_lm(last_words, states_aux_lm)[0])
                log_probs = (self.eta)*log_probs_tm + (1-self.eta)*log_probs_lm
            else:
                log_probs = log_probs_lm

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

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

            if self.score:
                new_score_lm = [[]] * n_samples
                new_score_tm = [[]] * n_samples

            #import ipdb; ipdb.set_trace()
            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                if self.enc_dec.state['include_lm']:
                    new_states_lm[i] = states_lm[orig_idx]
                if self.aux_lm:
                    new_states_aux_lm[i] = states_aux_lm[orig_idx]
                if self.score:
                    new_score_lm[i] = score_lm[orig_idx] + [log_probs_lm[orig_idx][next_word]]
                    new_score_tm[i] = score_tm[orig_idx] + [log_probs_tm[orig_idx][next_word]]
                inputs[i] = next_word

            if self.enc_dec.state['include_lm']:
                new_states, new_states_lm = self.comp_next_states(c, k, inputs, new_states_lm, *new_states)
            else:
                new_states = self.comp_next_states(c, k, inputs, new_states_lm, *new_states)

            # get previous hidden states of auxiliary language model
            if self.aux_lm:
                new_states_aux_lm = self.comp_next_states_lm(inputs, new_states_aux_lm)[0]

            if self.score:
                score_lm = []
                score_tm = []

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                    if self.score:
                        score_lm.append(new_score_lm[i])
                        score_tm.append(new_score_tm[i])
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    if self.score:
                        fin_score_lm.append(new_score_lm[i])
                        fin_score_tm.append(new_score_tm[i])

            if self.enc_dec.state['include_lm']:
                states = map(lambda x : x[indices], [new_states])
                states_lm = new_states_lm[indices]
            else:
                states = map(lambda x : x[indices], new_states)

            if self.aux_lm:
                states_aux_lm = new_states_aux_lm[indices]

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, False, minlen)
            else:
                logger.error("Translation failed")
        LM_score = numpy.array([numpy.sum(f,axis=0) for f in fin_score_lm])[numpy.argsort(fin_costs)]
        TM_score = numpy.array([numpy.sum(f,axis=0) for f in fin_score_tm])[numpy.argsort(fin_costs)]
        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_trans, fin_costs, LM_score, TM_score

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seq, n_samples,
        sampler=None, beam_search=None,
        ignore_unk=False, normalize=False,
        alpha=1, verbose=False):
    if beam_search:
        sentences = []
        trans, costs, score_lm, score_tm = beam_search.search(seq, n_samples,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {} - lmScore[{}] tmScore[{}]".format(costs[i],
                        sentences[i],score_lm[i],score_tm[i])
        return sentences, costs, trans
    elif sampler:
        sentences = []
        all_probs = []
        costs = []

        values, cond_probs = sampler(n_samples, 3 * (len(seq) - 1), alpha, seq)
        for sidx in xrange(n_samples):
            sen = []
            for k in xrange(values.shape[0]):
                if lm_model.word_indxs[values[k, sidx]] == '<eol>':
                    break
                sen.append(lm_model.word_indxs[values[k, sidx]])
            sentences.append(" ".join(sen))
            probs = cond_probs[:, sidx]
            probs = numpy.array(cond_probs[:len(sen) + 1, sidx])
            all_probs.append(numpy.exp(-probs))
            costs.append(-numpy.sum(probs))
        if normalize:
            counts = [len(s.strip().split(" ")) for s in sentences]
            costs = [co / cn for co, cn in zip(costs, counts)]
        sprobs = numpy.argsort(costs)
        if verbose:
            for pidx in sprobs:
                print "{}: {} {} {}".format(pidx, -costs[pidx], all_probs[pidx], sentences[pidx])
            print
        return sentences, costs, None
    else:
        raise Exception("I don't know what to do")


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
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
    parser.add_argument("model_path",
            help="Path to the model")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    parser.add_argument("--lm-state",
            default=None,
            help="State to use as an auxiliary LM")
    parser.add_argument("--lm-model",
            default=None,
            help="Model to use as an auxiliary LM")
    parser.add_argument("--eta",
            default=0.5, type=float,
            help="Balancing parameter between TM and LM log-probs")
    parser.add_argument("--score", action="store_true",
            default=False, help="Score translations by auxiliary language model")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))

    # employ language model with shallow fusion, this part is
    # partially unnecessary since it is only done to use load()
    # from model, another way can be inheriting from Container
    aux_lm_builder = None
    if args.lm_state and args.lm_model:
        logging.debug("Create auxiliary language model")
        aux_lm_state = prototype_lm_state()
        with open(args.lm_state) as lms:
            aux_lm_state.update(cPickle.load(lms))
        aux_lm_builder = LM_builder(aux_lm_state,rng,skip_init=True)
        aux_lm_builder.build()
        aux_lm_builder.lm_model = aux_lm_builder.create_lm_model()
        aux_lm_builder.lm_model.load(args.lm_model)

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec, aux_lm_builder, args.eta, args.score)
        beam_search.compile()
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'],'r'))
    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        start_time = time.time()

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            if state['source_encoding'] == 'utf8':
                seqin = line.strip().decode('utf-8')
            else:
                seqin = line.strip()

            seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
            if args.verbose:
                print "Parsed Input:", parsed_in
            trans, costs, _ = sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk,
                    normalize=args.normalize, verbose=args.verbose)
            best = numpy.argmin(costs)
            if state['target_encoding'] == 'utf8':
                print >>ftrans, trans[best].encode('utf8').replace(" ","")
            else:
                print >>ftrans, trans[best]
            if args.verbose:
                if state['target_encoding'] == 'utf8':
                    print "Translation:", trans[best].encode('utf8')
                else:
                    print "Translation:", trans[best]
            total_cost += costs[best]
            if (i + 1)  % 100 == 0:
                ftrans.flush()
                logger.debug("Current speed is {} per sentence".
                        format((time.time() - start_time) / (i + 1)))
        print "Total cost of the translations: {}".format(total_cost)

        fsrc.close()
        ftrans.close()
    else:
        while True:
            try:
                seqin = raw_input('Input Sequence: ')
                n_samples = int(raw_input('How many samples? '))
                alpha = None
                if not args.beam_search:
                    alpha = float(raw_input('Inverse Temperature? '))
                seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print "Parsed Input:", parsed_in
            except Exception:
                print "Exception while parsing your input:"
                traceback.print_exc()
                continue

            sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search,
                    ignore_unk=args.ignore_unk, normalize=args.normalize,
                    alpha=alpha, verbose=True)

if __name__ == "__main__":
    main()
