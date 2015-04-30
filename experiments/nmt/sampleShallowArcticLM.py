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
    parse_input

import lstmlm as arcLM
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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

    def __init__(self, enc_dec, aux_lm=None, eta=None, score=False,
                 weight_lm_only=False):
        self.enc_dec = enc_dec
        self.aux_lm = aux_lm
        self.eta = eta
        self.score = score
        self.weight_lm_only = weight_lm_only

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

        # compile sampling and prob functions if necessary
        if self.aux_lm:
            self.aux_lm_f_next = arcLM.build_sampler(self.aux_lm['model_tparams'],
                                                     self.aux_lm['model_options'],
                                                     RandomStreams(1234))

    def search(self, seq, n_samples, ignore_unk=False,
               minlen=1, eos_id=0, unk_id=1, cross_dict=None):
        c = self.comp_repr(seq)[0]
        states = map(lambda x: x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        # Set initial states of  the language
        # model inside of the decoder
        states_lm = None
        states_mem_lm = None
        if self.enc_dec.state['include_lm']:
            dim_lm = self.enc_dec.decoder.state['lm_readout_dim']
            states_lm = numpy.zeros((1, dim_lm), dtype="float32")
            if self.enc_dec.state['use_arctic_lm']:
                states_mem_lm = numpy.zeros((1, dim_lm), dtype="float32")

        # Set initial states of the auxiliary
        # language mode, independent of decoder
        states_aux_lm = None
        memory_aux_lm = None
        if self.aux_lm:
            dim_aux_lm = self.aux_lm['model_options']['dim']
            states_aux_lm = numpy.zeros((1, dim_aux_lm), dtype="float32")
            if self.aux_lm['model_options']['rec_layer'] == 'lstm':
                memory_aux_lm = numpy.zeros((1, dim_aux_lm)).astype('float32')

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
            new_states_mem_lm = None
            if self.enc_dec.state['include_lm']:
                new_states_lm = numpy.zeros((n_samples, dim_lm), dtype="float32")
                if self.enc_dec.state['use_arctic_lm']:
                    new_states_mem_lm = numpy.zeros((n_samples, dim_lm), dtype="float32")


            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t: t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))

            # reset states of the auxiliary language model
            new_states_aux_lm = None
            new_memory_aux_lm = None
            if self.aux_lm:
                new_states_aux_lm = numpy.zeros((n_samples, dim_aux_lm), dtype="float32")
                if self.aux_lm['model_options']['rec_layer'] == 'lstm':
                    new_memory_aux_lm = numpy.zeros((n_samples, dim_aux_lm), dtype="float32")

                # If TM and LM dictionaries are not matching use a cross
                # dictionary for the correct mapping
                last_words_lm = last_words
                if cross_dict and k > 0:
                    last_words_lm = numpy.asarray([cross_dict[el]
                                                   for el in last_words])

            log_probs_tm = numpy.log(self.comp_next_probs(
                                     c, k, last_words, cross_dict,
                                     states_lm, states_mem_lm, *states)[0])

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs_tm[:, unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs_tm[:, eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs_tm = numpy.array(costs)[:, None] - log_probs_tm
            flat_next_costs_tm = next_costs_tm.flatten()
            best_costs_indices_tm = argpartition(
                    flat_next_costs_tm.flatten(),
                    n_samples)[:n_samples]

            # get log probability given last words and previous hidden states
            # and then fuse it with TM log probability by their geometric mean
            if self.aux_lm:
                probs_lm, _, _, _ = self.aux_lm_f_next(last_words_lm,
                    states_aux_lm, memory_aux_lm)

                # append last column for eos symbol - which is excluded in arctic
                #probs_lm = numpy.append(probs_lm, probs_lm[:,-1][:,None], axis=1)

                # set prev last column to zero
                #probs_lm[:,-2] = 0.

                # set unk probability to zero and re-normalize whole distribution
                # and convert it back to log-probability. Then set the eos probability
                # of language model to eos probability of translation model
                unk_id_lm = self.aux_lm['model_options']['unk_idx']
                eos_id_lm = self.aux_lm['model_options']['n_words'] - 1
                probs_lm[:, unk_id_lm] = 1e-16
                probs_lm = probs_lm / probs_lm.sum(axis=1)[:, None]

                log_probs_lm = numpy.log(probs_lm)

                #log_probs_lm[numpy.arange(last_words.shape[0]),self.enc_dec.state['null_sym_target']] = \
                #            log_probs_tm[numpy.arange(last_words.shape[0]),self.enc_dec.state['null_sym_target']]

                flat_next_costs_lm = - log_probs_lm.flatten()

            # Decypher flatten indices
            voc_size = log_probs_tm.shape[1]
            trans_indices = best_costs_indices_tm / voc_size
            word_indices = best_costs_indices_tm % voc_size

            # get the geometric mean here
            if self.aux_lm:
                best_costs_indices_lm = numpy.zeros_like(best_costs_indices_tm)
                if cross_dict:
                    voc_size_lm = self.aux_lm['model_options']['n_words']
                    for cid, (tid, wid) in enumerate(zip(trans_indices, word_indices)):
                        wid_lm = cross_dict[wid]
                        best_costs_indices_lm[cid] = wid_lm + (tid * voc_size_lm)

                if self.weight_lm_only:
                    costs = [flat_next_costs_tm[ww] +
                             (1-self.eta) * flat_next_costs_lm[zz]
                             if (zz != eos_id_lm and zz != unk_id_lm)
                             else flat_next_costs_tm[ww]
                             for ww, zz in zip(best_costs_indices_tm,
                                               best_costs_indices_lm)]
                else:
                    costs = [(self.eta) * flat_next_costs_tm[ww] +
                             (1-self.eta) * flat_next_costs_lm[zz]
                             if (zz != eos_id_lm and zz != unk_id_lm)
                             else flat_next_costs_tm[ww]
                             for ww, zz in zip(best_costs_indices_tm,
                                               best_costs_indices_lm)]
            else:
                    costs = flat_next_costs_tm[best_costs_indices_tm]

            if self.score:
                new_score_tm = [[]] * n_samples
                if self.aux_lm:
                    new_score_lm = [[]] * n_samples

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
                    if self.enc_dec.state['use_arctic_lm']:
                        new_states_mem_lm[i] = states_mem_lm[orig_idx]

                if self.aux_lm:
                    new_states_aux_lm[i] = states_aux_lm[orig_idx]
                    new_memory_aux_lm[i] = memory_aux_lm[orig_idx]

                if self.score:
                    new_score_tm[i] = score_tm[orig_idx] + [log_probs_tm[orig_idx][next_word]]
                    if self.aux_lm:
                        if cross_dict:
                            next_word_lm = cross_dict[next_word]
                            new_score_lm[i] = score_lm[orig_idx] + \
                                [log_probs_lm[orig_idx][next_word_lm]]
                        else:
                            new_score_lm[i] = score_lm[orig_idx] + \
                                [log_probs_lm[orig_idx][next_word]]

                inputs[i] = next_word

            if self.enc_dec.state['include_lm']:
                if self.enc_dec.state['use_arctic_lm']:
                    new_states, new_states_lm, new_states_mem_lm = self.comp_next_states(
                            c, k, inputs, cross_dict,
                            new_states_lm, new_states_mem_lm, *new_states)
                else:
                    new_states, new_states_lm = self.comp_next_states(
                            c, k, inputs, cross_dict,
                            new_states_lm, new_states_mem_lm, *new_states)
            else:
                new_states = self.comp_next_states(c, k, inputs, cross_dict,
                                                   new_states_lm,
                                                   new_states_mem_lm,
                                                   *new_states)

            # get previous hidden states of auxiliary language model
            if self.aux_lm:
                inputs_aux_lm = inputs
                if cross_dict:
                    inputs_aux_lm = numpy.asarray([cross_dict[xx]
                                                   for xx in inputs_aux_lm])

                # Currently this is being done in cross_dict
                #inputs_aux_lm[numpy.where(inputs==self.enc_dec.state['null_sym_target'])]=\
                #        self.enc_dec.state['null_sym_target']-1

                _, _,\
                new_states_aux_lm,\
                new_memory_aux_lm = self.aux_lm_f_next(inputs_aux_lm,
                                                       new_states_aux_lm,
                                                       new_memory_aux_lm)
                if k < 2:
                    new_states_aux_lm = 0. * new_states_aux_lm
                    new_memory_aux_lm = 0. * new_memory_aux_lm

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
                        if self.aux_lm:
                            score_lm.append(new_score_lm[i])
                        score_tm.append(new_score_tm[i])
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    if self.score:
                        if self.aux_lm:
                            fin_score_lm.append(new_score_lm[i])
                        fin_score_tm.append(new_score_tm[i])

            if self.enc_dec.state['include_lm']:
                states = map(lambda x: x[indices], [new_states])
                states_lm = new_states_lm[indices]
                if self.enc_dec.state['use_arctic_lm']:
                    states_mem_lm = new_states_mem_lm[indices]
            else:
                states = map(lambda x: x[indices], new_states)

            if self.aux_lm:
                states_aux_lm = new_states_aux_lm[indices]
                memory_aux_lm = new_memory_aux_lm[indices]

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
        LM_score = numpy.zeros(len(fin_costs))
        TM_score = numpy.zeros(len(fin_costs))
        if self.score:
            if self.aux_lm:
                LM_score = numpy.array(
                    [numpy.sum(f, axis=0)
                     for f in fin_score_lm])[numpy.argsort(fin_costs)]
            TM_score = numpy.array(
                [numpy.sum(f, axis=0)
                 for f in fin_score_tm])[numpy.argsort(fin_costs)]
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
           alpha=1, verbose=False, unk_id=1, eos_id=0,
           cross_dict=None):
    if beam_search:
        sentences = []
        trans, costs, score_lm, score_tm = beam_search.search(
            seq, n_samples, cross_dict=cross_dict,
            ignore_unk=ignore_unk, minlen=len(seq) / 2,
            unk_id=unk_id, eos_id=eos_id)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {} - lmScore[{}] tmScore[{}]".format(
                    costs[i], sentences[i], score_lm[i], score_tm[i])
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
    parser.add_argument("--model_path",
            required=True, help="Path to the model")
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
    parser.add_argument("changes",
            nargs="*", default="",
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
    parser.add_argument("--n-best",
            default=1, type=int,
            help="N-best list results to be written to a separate file")
    parser.add_argument("--score", action="store_true",
            default=False, help="Score translations by auxiliary language model")
    parser.add_argument("--weight-lm-only",
            action="store_true", default=False,
            help="use eta * prob_tm or not")
    parser.add_argument("--cross-dict", action="store_true", default=False,
                        help="Build cross dictionary from TM indices to LM indices")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("{%s}" % args.changes[0]))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))

    # employ language model with shallow fusion
    aux_lm = {}
    if args.lm_state and args.lm_model:
        logging.debug("Create auxiliary language model from arctic")

        # load language model parameters
        aux_lm['model_options'] = arcLM.load_options(args.lm_model)

        # update it with changes
        #aux_lm.update(eval("{%s}" % args.changes[0]))

        # load and convert theano parameters
        aux_lm['model_tparams'] = arcLM.init_tparams(arcLM.load_params(args.lm_model,
                                           arcLM.init_params(aux_lm['model_options'])))

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec, aux_lm, args.eta,
                score=True, weight_lm_only=args.weight_lm_only)
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
        if args.n_best > 0:
            fidx_scores = open(args.trans + '_SCORES', 'w')
        start_time = time.time()

        n_samples = args.beam_size

        # Build cross dictionary
        cross_dict = None
        if aux_lm and args.cross_dict:
            cross_dict = {}
            for idx, word in lm_model.word_indxs.iteritems():
                cross_dict[idx] = aux_lm['dict'].get(
                        word, aux_lm['model_options']['unk_idx'])
            cross_dict[eos_id] = aux_lm['model_options']['n_words'] - 1
            cross_dict[unk_id] = aux_lm['model_options']['unk_idx']

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
                                     normalize=args.normalize,
                                     cross_dict=cross_dict,
                                     eos_id=eos_id,
                                     unk_id=unk_id,
                                     verbose=args.verbose)
            nbest_idx = numpy.argsort(costs)[:args.n_best]
            for j, best in enumerate(nbest_idx):
                if state['target_encoding'] == 'utf8':
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
