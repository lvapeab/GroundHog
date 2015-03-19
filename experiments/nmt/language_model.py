#!/usr/bin/env python

import argparse
import numpy
import logging
import pprint
import operator
import itertools

import cPickle

import theano
import theano.tensor as TT
from theano import scan
from theano.sandbox.scan import scan as scan_sandbox
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.datasets import PytablesBitextIterator_UL
from groundhog.models import LM_Model
from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD_rmspropv2 import SGD as SGD_rmsprop
from groundhog.mainLoop import MainLoop

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecurrentMultiLayer, \
        RecurrentMultiLayerInp, \
        RecurrentMultiLayerShortPath, \
        RecurrentMultiLayerShortPathInp, \
        RecurrentMultiLayerShortPathInpAll, \
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate

import experiments.nmt

def none_if_zero(x):
    if x == 0:
        return None
    return x

def create_padded_batch(state, x, y, seqlen=None, return_dict=False):
    """
    A callback given to the iterator to transform data in suitable format

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
    """

    if seqlen is None:
        seqlen = state['seqlen']

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

    # We say that an input pair is valid if both:
    # - either source sequence or target sequence is non-empty
    # - source sequence and target sequence have null_sym ending
    # Why did not we filter them earlier?
    for idx in xrange(X.shape[1]):
        if numpy.sum(Xmask[:,idx]) == 0 and numpy.sum(Ymask[:,idx]) == 0:
            null_inputs[idx] = 1
        if Xmask[-1,idx] and X[-1,idx] != state['null_sym']:
            null_inputs[idx] = 1
        if Ymask[-1,idx] and Y[-1,idx] != state['null_sym']:
            null_inputs[idx] = 1

    valid_inputs = 1. - null_inputs

    # Leave only valid inputs
    X = X[:,valid_inputs.nonzero()[0]]
    Y = Y[:,valid_inputs.nonzero()[0]]
    Xmask = Xmask[:,valid_inputs.nonzero()[0]]
    Ymask = Ymask[:,valid_inputs.nonzero()[0]]
    if len(valid_inputs.nonzero()[0]) <= 0:
        return None

    # Unknown words
    X[X >= state['n_sym']] = state['unk_sym']
    Y[Y >= state['n_sym']] = state['unk_sym']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask

def get_batch_iterator(state):
    """
    get_batch_iterator returns an iterator that respects
    the standard python iterator protocol, it has options to
    allow for infinite looping for a training set, or finite
    looping for a validation set

    The Iterator class defined inheriets from a PytablesBitextIterator
    or some variant, which manages the PytablesBitextFetcher class
    which iterfaces with the HDF5 file, using another thread to
    reduce computations spent shuttling from the disc

    The Iterator object at one time only loads in k_batches into memory,
    then proprocesses the batch by adding masks to the data before
    returning.

    Kelvin Xu
    """
    class Iterator(PytablesBitextIterator_UL):

        def __init__(self, mode='train', val_size=1000, *args, **kwargs):
            PytablesBitextIterator_UL.__init__(self, *args, **kwargs)
            self.mode = mode
            self.val_size = val_size
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator_UL.next(self) for k in range(k_batches)]
                x = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
                y = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
                lens = numpy.asarray([map(len, x), map(len, y)])
                order = numpy.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else numpy.arange(len(x))
                for k in range(k_batches):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]], [y[indices]],
                            return_dict=True)
                    if batch:
                        yield batch

        def next(self, peek=False):

            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()

            if self.peeked_batch:
                # Only allow to peek one batch
                assert not peek
                logger.debug("Use peeked batch")
                batch = self.peeked_batch
                self.peeked_batch = None
                return batch

            if not self.batch_iter:
                raise StopIteration

            batch = next(self.batch_iter)
            if peek:
                self.peeked_batch = batch
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        target_file=state['target'],
        can_fit=False,
        queue_size=1000,
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        n_words=state['n_sym']
    )

    valid_data = PytablesBitextIterator_UL(
        batch_size=int(state['val_bs']),
        val_size=887,
        target_file=state['val_target'],
        can_fit=True,
        shuffle=False,
        use_infinite_loop=True,
        n_words=state['n_sym'],
        mode='valid',
        state=state)

    return train_data, valid_data


class LM_builder(object):
    """
    Object that encapsulates the languages model
    """

    def __init__(self, state, rng, skip_init=False):
        self.state = state
        self.rng = rng
        self.skip_init = skip_init

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self.__create_layers__()

    def __create_layers__(self, build_output=True):

        self.emb_words = MultiLayer(
            self.rng,
            n_in=self.state['n_sym'],
            n_hids=self.state['rank_n_approx'],
            activation=eval(self.state['rank_n_activ']),
            learn_bias = True,
            bias_scale=self.state['bias'],
            name='lm_emb_words',
            **self.default_kwargs)

        self.rec = eval(self.state['rec_layer'])(
                self.rng,
                n_hids=self.state['dim'],
                activation = eval(self.state['activ']),
                bias_scale = self.state['bias'],
                scale=self.state['rec_weight_scale'],
                init_fn=self.state['rec_weight_init_fn']
                    if not self.skip_init
                    else "sample_zeros",
                weight_noise=self.state['weight_noise_rec'],
                gating=self.state['rec_gating'],
                gater_activation=self.state['rec_gater'],
                reseting=self.state['rec_reseting'],
                reseter_activation=self.state['rec_reseter'],
                name='lm_rec')

        self.pre_word_readout = MultiLayer(self.rng,
                                            n_in=self.state['rank_n_approx'],
                                            n_hids=self.state['dim'],
                                            activation=eval(self.state['activ']),
                                            learn_bias=True,
                                            bias_scale=self.state['bias'],
                                            name='lm_bigram_prev_word',
                                            **self.default_kwargs)

        gate_kwargs = dict(self.default_kwargs)

        gate_kwargs.update(dict(
            n_in=self.state['rank_n_approx'],
            n_hids=[self.state['dim']],
            activation=['lambda x:x']))

        self.inputer = MultiLayer(
                self.rng,
                name='inputer',
                **gate_kwargs)

        self.reseter = MultiLayer(
                self.rng,
                name='reseter',
                **gate_kwargs)

        self.updater = MultiLayer(
                self.rng,
                name='updater',
                **gate_kwargs)

        # if all we care about is the hidden layer
        if build_output:
            self.output_layer = SoftmaxLayer(self.rng,
                                             self.state['dim'],
                                             self.state['n_sym'],
                                             self.state['out_scale'],
                                             self.state['out_sparse'],
                                             init_fn="sample_weights_classic",
                                             weight_noise=self.state['weight_noise'],
                                             sum_over_time=True,
                                             name='lm_out')

    def build_for_translation(self, target, target_mask=None, prev_hid=None):
        """
        This function builds the language model for the machine translation
        system, naturally it requires the target, and target_mask

        In the sampling step, we have no target_mask

        """
        # TODO
        # this assertion doesn't really make sense at the moment.
        #assert target.ndim == 3
        if target.ndim == 1:
            n_samples = 1
        else:
            n_samples = target.shape[1]

        x_emb = self.emb_words(target, no_noise_bias=self.state['no_noise_bias'])

        input_signals = self.inputer(x_emb)
        update_signals = self.updater(x_emb)
        reset_signals = self.reseter(x_emb)

        # if we are doing sampling, we expect a prev_hid state
        if prev_hid is not None:
            self.rec_layer = self.rec(input_signals,
                        state_before=prev_hid,
                        no_noise_bias=self.state['no_noise_bias'],
                        one_step=True,
                        use_noise=False,
                        gater_below=none_if_zero(update_signals),
                        reseter_below=none_if_zero(reset_signals),
                         )
        else:
            self.rec_layer = self.rec(input_signals, mask=target_mask,
                        no_noise_bias=self.state['no_noise_bias'],
                        batch_size=n_samples,
                        one_step=False,
                        gater_below=none_if_zero(update_signals),
                        reseter_below=none_if_zero(reset_signals),
                        )
        return self.rec_layer

    def get_const_params(self):
        """
        This function grabs the params to be excluded from
        gradient calculation
        """
        const_params = []
        # embedding of language model
        const_params += self.emb_words.params
        # Gating units
        const_params += self.inputer.params
        const_params += self.updater.params
        const_params += self.reseter.params
        cost_params += self.pre_word_readout.params

        # Recurrent units
        const_params += self.rec.params
        return const_params

    # TODO build sampler for translation
    def get_sampler_translation(self):
        pass

    def build(self, build_output=True):
        """
        Build Computational Graph
        """
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')

        n_samples = self.x.shape[1]

        self.inputs = [self.x, self.y, self.x_mask, self.y_mask]

        # the demensions of this is (time * batch_id, embedding_dim)
        # the whole input is flattened to support advanced indexing < -- should read this.
        #
        self.x_emb = self.emb_words(self.x, no_noise_bias=self.state['no_noise_bias'])

        x_input = self.inputer(self.x_emb)
        update_signals = self.updater(self.x_emb)
        reset_signals = self.reseter(self.x_emb)

        self.rec_layer = self.rec(x_input, mask=self.x_mask,
                    no_noise_bias=self.state['no_noise_bias'],
                    batch_size=n_samples,
                    gater_below=none_if_zero(update_signals),
                    reseter_below=none_if_zero(reset_signals),
                     )

        self.readout = self.rec_layer

        if self.state['bigram']:
            check_first_word = (self.y > 0
                        if self.state['check_first_word']
                        else TT.ones((y.shape[0]), dtype="float32"))

            AA = TT.shape_padright(check_first_word) * self.pre_word_readout(self.x_emb)
            self.readout = self.rec_layer + AA

        self.train_model = self.output_layer(self.readout).train(target=self.y,
                                                                 mask=self.y_mask)

    def build_validation(self, build_output=True):

        #if self.state['val_bs'] == 1:
        #    self.vx = TT.lvector('x')
        #    self.vy = TT.lvector('y')
        #    self.vx_mask = TT.lvector('x_mask')
        #    self.vy_mask = TT.lvector('y_mask')
        #    n_samples = 1
        #else:
        self.vx = TT.lmatrix('x')
        self.vy = TT.lmatrix('y')
        self.vx_mask = TT.matrix('x_mask')
        self.vy_mask = TT.matrix('y_mask')
        n_samples = self.vx.shape[1]

        if "debug" in state and state['debug']:
            theano.config.compute_test_value = 'warn'
            self.vx.tag.test_value = numpy.arange(100).reshape((50, -1)).astype("int64")
            self.vy.tag.test_value = numpy.arange(100).reshape((50, -1)).astype("int64")
            self.vx_mask.tag.test_value = numpy.zeros(100).reshape((50, -1)).astype("int64")
            self.vy_mask.tag.test_value = numpy.zeros(100).reshape((50, -1)).astype("int64")

        self.val_inputs = [self.vx, self.vy, self.vx_mask, self.vy_mask]
        self.vx_emb= self.emb_words(self.vx, no_noise_bias=self.state['no_noise_bias'])
        vx_input = self.inputer(self.vx_emb)
        update_signals = self.updater(self.vx_emb)
        reset_signals = self.reseter(self.vx_emb)

        vrec_layer = self.rec(vx_input,
                              mask=self.vx_mask,
                              no_noise_bias=self.state['no_noise_bias'],
                              batch_size=n_samples,
                              gater_below=none_if_zero(update_signals),
                              reseter_below=none_if_zero(reset_signals))

        vreadout = vrec_layer
        if state['bigram']:
            if self.vy.ndim == 1:
                vreadout += Shift()(self.pre_word_readout(self.vx_emb).reshape((self.vy.shape[0], 1, self.state['dim'])))
            else:
                # This place needs explanation. When prev_word_readout is applied to
                # approx_embeddings the resulting shape is
                # (n_batches * sequence_length, repr_dimensionality). We first
                # transform it into 3D tensor to shift forward in time. Then
                # reshape it back.
                vreadout += Shift()(self.pre_word_readout(self.vx_emb).reshape((self.vy.shape[0],
                                                                                 self.vy.shape[1],
                                                                                 self.state['dim']))).reshape(vrec_layer.out.shape)

        self.valid_model = self.output_layer(vreadout, use_noise=False).validate(target=self.vy,
                                                                                   mask=self.vy_mask,
                                                                                   sum_over_time=True)

    def get_valid_fn(self):
        assert self.valid_model is not None, "Validation function should not be empty."
        valid_fn = theano.function(self.val_inputs,
                                   self.valid_model.cost,
                                   name="valid_fn",
                                   updates=[])
        return valid_fn

    def get_sampler(self):
        """
        Sampling
        """

        # TODO change sampler so only return h0

        def sample_fn(word_tm1, h_tm1):
            x_emb = self.emb_words(word_tm1, use_noise = False, one_step=True)
            x_input = self.inputer(x_emb)
            update_signal = self.updater(x_emb)
            reset_signal = self.reseter(x_emb)
            h0 = self.rec(x_input, gater_below=update_signal,
                          reseter_below=reset_signal,
                          state_before=h_tm1,
                          one_step=True, use_noise=False)
            word = self.output_layer.get_sample(state_below=h0)
            return word, h0

        # oddly non-functional scan code
        #[samples, summaries], updates = scan(sample_fn,
        #                    outputs_info=[TT.alloc(numpy.int64(5),1),
        #                    TT.alloc(numpy.float32(0), 1,state['dim'])
        #                    ],
        #                    n_steps= state['sample_steps'],
        #                    name='sampler_scan')

        ##### scan for iterating the single-step sampling multiple times
        [samples, summaries], updates = scan_sandbox(sample_fn,
                          states = [
                              TT.alloc(numpy.int64(state['sampling_seed']), state['sample_steps']),
                              TT.alloc(numpy.float32(0), 1, state['dim'])],
                          n_steps= state['sample_steps'],
                          name='sampler_scan')

        ##### define a Theano function
        sample_fn = theano.function([], [samples],
                updates=updates, profile=False, name='sample_fn')

        return sample_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

if __name__ == '__main__':
    # copying construction from experiment/nmt/train.py
    # Grab prototype specified by proto
    args = parse_args()
    state = getattr(experiments.nmt, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    rng = numpy.random.RandomState(state['seed'])

    train_data, valid_data = get_batch_iterator(state)

    #valid_data.start()
    model = LM_builder(state, rng, skip_init=True
                       if state['reload'] else False)

    model.build()
    model.build_validation()

    #TODO
    valid_fn = model.get_valid_fn()

    lm_model = LM_Model(
        cost_layer = model.train_model,
        weight_noise_amount=state['weight_noise_amount'],
        valid_fn = valid_fn,
        indx_word=state['indx_word'],
        indx_word_src=state['indx_word'],
        clean_before_noise_fn = False,
        noise_fn = None,
        rng = rng)

    algo = eval(state['algo'])(lm_model, state, train_data)

    sampler = model.get_sampler()
    def hook_fn():
        idict = cPickle.load(open(state['indx_word'], 'r'))
        idict[state['unk_sym']]='<unk>'
        idict[state['null_sym']]='<eos>'
        sample = sampler()[0]
        # prepend the seed
        sample = numpy.insert(sample, 0, state['sampling_seed'])
        print " ".join([idict[si] for si in sample])

    main = MainLoop(train_data,
                    valid_data,
                    None, lm_model,
                    algo, state, None,
                    reset=state['reset'],
                    hooks= hook_fn
                    if state['hookFreq'] >= 0
                    else None)

    if state['reload']:
        main.load()

    if state['loopIters'] > 0:
        main.main()


