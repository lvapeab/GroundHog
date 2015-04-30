import numpy
import logging
import pprint
import operator
import itertools
import copy
import theano
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.layers import \
                        Layer, \
                        MultiLayer, \
                        SoftmaxLayer, \
                        HierarchicalSoftmaxLayer, \
                        LSTMLayer, \
                        RecurrentLayer, \
                        DoubleRecurrentLayer, \
                        RecursiveConvolutionalLayer, \
                        UnaryOp, \
                        Shift, \
                        LastState, \
                        DropOp, \
                        OneLayer, \
                        Concatenate

from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator
from groundhog.utils import (sample_zeros, \
                             sample_weights_orth, \
                             init_bias, \
                             sample_weights_classic,
                             name2pos)

import groundhog.utils as utils

from experiments.nmt import (LM_builder, \
                             prototype_lm_state, \
                             prototype_lm_state_en, \
                             prototype_lm_state_en_finetune, \
                             prototype_lm_state_en_finetune_union)

import lstmlm
from lstmlm.lm_barebone import (lstm_layer, \
                                build_layer_tm, \
                                build_samp_layer_tm, \
                                build_model, \
                                get_layer_graph, \
                                get_model_graph)

logger = logging.getLogger(__name__)

def create_padded_batch(state, x, y, return_dict=False):
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
    """

    mx = state['seqlen']
    my = state['seqlen']
    if state['trim_batches']:
        # Similar length for all source sequences
        mx = numpy.minimum(state['seqlen'], max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = numpy.minimum(state['seqlen'], max([len(xx) for xx in y[0]]))+1

    # Batch size
    n = x[0].shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        if mx < len(x[0][idx]):
            X[:mx, idx] = x[0][idx][:mx]
        else:
            X[:len(x[0][idx]), idx] = x[0][idx][:mx]

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[len(x[0][idx]):, idx] = state['null_sym_source']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[0][idx]), idx] = 1.
        if len(x[0][idx]) < mx:
            Xmask[len(x[0][idx]), idx] = 1.

    # Fill Y and Ymask in the same way as X and Xmask in the previous loop
    for idx in xrange(len(y[0])):
        Y[:len(y[0][idx]), idx] = y[0][idx][:my]
        if len(y[0][idx]) < my:
            Y[len(y[0][idx]):, idx] = state['null_sym_target']
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
        if Xmask[-1,idx] and X[-1,idx] != state['null_sym_source']:
            null_inputs[idx] = 1
        if Ymask[-1,idx] and Y[-1,idx] != state['null_sym_target']:
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
    X[X >= state['n_sym_source']] = state['unk_sym_source']
    Y[Y >= state['n_sym_target']] = state['unk_sym_target']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask

def get_batch_iterator(state):

    class Iterator(PytablesBitextIterator):

        def __init__(self, *args, **kwargs):
            PytablesBitextIterator.__init__(self, *args, **kwargs)
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator.next(self) for k in range(k_batches)]
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
        target_file=state['target'][0],
        source_file=state['source'][0],
        can_fit=False,
        queue_size=1000,
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        max_len=state['seqlen'])
    return train_data

class RecurrentLayerWithSearch(Layer):
    """A copy of RecurrentLayer from groundhog"""

    def __init__(self, rng,
                 n_hids,
                 c_dim=None,
                 scale=.01,
                 activation=TT.tanh,
                 bias_fn='init_bias',
                 bias_scale=0.,
                 init_fn='sample_weights',
                 gating=False,
                 reseting=False,
                 dropout=1.,
                 gater_activation=TT.nnet.sigmoid,
                 reseter_activation=TT.nnet.sigmoid,
                 weight_noise=False,
                 deep_attention=None,
                 deep_attention_n_hids=None,
                 deep_attention_acts=None,
                 name=None):
        logger.debug("RecurrentLayerWithSearch is used")

        self.grad_scale = 1
        assert gating == True
        assert reseting == True
        assert dropout == 1.
        assert weight_noise == False
        updater_activation = gater_activation

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(updater_activation) is str or type(updater_activation) is unicode:
            updater_activation = eval(updater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)

        self.scale = scale
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.updater_activation = updater_activation
        self.reseter_activation = reseter_activation
        self.c_dim = c_dim
        self.deep_attention = deep_attention
        self.deep_attention_n_hids = deep_attention_n_hids
        self.deep_attention_acts = deep_attention_acts

        assert rng is not None, "random number generator should not be empty!"

        super(RecurrentLayerWithSearch, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.params = []
        self._init_params()

    def _init_params(self):

        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                -1,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]

        self.G_hh = theano.shared(
                self.init_fn(self.n_hids,
                    self.n_hids,
                    -1,
                    self.scale,
                    rng=self.rng),
                name="G_%s"%self.name)
        self.params.append(self.G_hh)

        self.R_hh = theano.shared(
                self.init_fn(self.n_hids,
                    self.n_hids,
                    -1,
                    self.scale,
                    rng=self.rng),
                name="R_%s"%self.name)
        self.params.append(self.R_hh)

        self.A_cp = theano.shared(
                sample_weights_classic(self.c_dim,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="A_%s"%self.name)
        self.params.append(self.A_cp)

        self.B_hp = theano.shared(
                sample_weights_classic(self.n_hids,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="B_%s"%self.name)
        self.params.append(self.B_hp)

        self.D_pe = theano.shared(
                numpy.zeros((self.n_hids, 1), dtype="float32"),
                name="D_%s"%self.name)

        self.params.append(self.D_pe)

        if self.deep_attention:
            self.DatN = MultiLayer(rng=self.rng,
                                   n_in=self.n_hid, # birnn enc-hids + dec-hid
                                   n_hids=self.deep_attention_n_hids,
                                   activation=self.deep_attention_acts,
                                   name="DatN_%s"%self.name)

            [self.params.append(param) for param in self.DatN.params]

        self.params_grad_scale = [self.grad_scale for x in self.params]

    def set_decoding_layers(self, c_inputer, c_reseter, c_updater):
        self.c_inputer = c_inputer
        self.c_reseter = c_reseter
        self.c_updater = c_updater
        for layer in [c_inputer, c_reseter, c_updater]:
            self.params += layer.params
            self.params_grad_scale += layer.params_grad_scale

    def step_fprop(self,
                   state_below,
                   state_before,
                   gater_below=None,
                   reseter_below=None,
                   updater_below=None,
                   mask=None,
                   c=None,
                   c_mask=None,
                   p_from_c=None,
                   W_hh_=None,
                   G_hh_=None,
                   R_hh_=None,
                   A_cp_=None,
                   B_hp_=None,
                   D_pe_=None,
                   C_iW=None,
                   C_ib=None,
                   C_uW=None,
                   C_ub=None,
                   C_rW=None,
                   C_rb=None,
                   use_noise=True,
                   no_noise_bias=False,
                   step_num=None,
                   return_alignment=False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type updater_below: theano variable
        :param updater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        updater_below = gater_below
        if W_hh_ is not None:
            W_hh = W_hh_
        else:
            W_hh = self.W_hh

        if G_hh_ is not None:
            G_hh = G_hh_
        else:
            G_hh = self.G_hh

        if R_hh_ is not None:
            R_hh = R_hh_
        else:
            R_hh = self.R_hh

        if A_cp_ is not None:
            A_cp = A_cp_
        else:
            A_cp = self.A_cp

        if B_hp_ is not None:
            B_hp = B_hp_
        else:
            B_hp = self.B_hp

        if D_pe_ is not None:
            D_pe = D_pe_
        else:
            D_pe = self.D_pe

        # The code works only with 3D tensors
        cndim = c.ndim
        if cndim == 2:
            c = c[:, None, :]

        # Warning: either source_num or target_num should be equal,
        #          or one of them should be 1 (they have to broadcast)
        #          for the following code to make any sense.
        source_len = c.shape[0]
        source_num = c.shape[1]
        target_num = state_before.shape[0]
        dim = self.n_hids

        # Form projection to the tanh layer from the previous hidden state
        # Shape: (source_len, target_num, dim)
        p_from_h = ReplicateLayer(source_len)(utils.dot(state_before, B_hp)).out

        # Form projection to the tanh layer from the source annotation.
        if not p_from_c:
            p_from_c = utils.dot(c, A_cp).reshape((source_len, source_num, dim))

        # Sum projections - broadcasting happens at the dimension 1.
        joint_p = p_from_h + p_from_c

        # Apply deep MLP for deep attention
        if self.deep_attention:
            pp = self.DatN.fprop(joint_p)
        else:
            pp = joint_p

        # Apply non-linearity and project to energy.
        energy = TT.exp(utils.dot(TT.tanh(pp), D_pe)).reshape((source_len, target_num))
        if c_mask:
            # This is used for batches only, that is target_num == source_num
            energy *= c_mask

        # Calculate energy sums.
        normalizer = energy.sum(axis=0)

        # Get probabilities.:w
        probs = energy / normalizer

        # Calculate weighted sums of source annotations.
        # If target_num == 1, c shoulds broadcasted at the 1st dimension.
        # Probabilities are broadcasted at the 2nd dimension.
        ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)

        state_below += self.c_inputer(ctx).out
        reseter_below += self.c_reseter(ctx).out
        updater_below += self.c_updater(ctx).out

        # Reset gate:
        # optionally reset the hidden state.
        reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                reseter_below)
        reseted_state_before = reseter * state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        updater = self.updater_activation(TT.dot(state_before, G_hh) +
                                          updater_below)

        h = updater * h + (1 - updater) * state_before

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0, 'x')
            h = mask * h + (1 - mask) * state_before

        results = [h, ctx]
        if return_alignment:
            results += [probs]
        return results

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              c=None,
              c_mask=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias=False,
              return_alignment=False):

        updater_below = gater_below

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if updater_below:
                updater_below = updater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)
        p_from_c =  utils.dot(c, self.A_cp).reshape(
                (c.shape[0], c.shape[1], self.n_hids))

        non_seqs = [self.W_hh, self.G_hh, self.R_hh, self.A_cp, self.B_hp, self.D_pe,
                    updater_below, self.c_inputer.params[0],  self.c_inputer.params[1], self.c_updater.params[0],
                    self.c_updater.params[1], self.c_reseter.params[0], self.c_reseter.params[1]]
        if mask:
            sequences = [state_below, mask, updater_below, reseter_below]
            non_sequences = [c, c_mask, p_from_c]
            fn = lambda x, m, g, r,   h,   c1, cm, pc, whh, ghh, rhh,acp,bhp,dpe,ub,ciw,cib,cuw,cub,crw,crb : self.step_fprop(x, h, mask=m,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc, c_mask=cm,
                    W_hh_=whh, G_hh_=ghh, R_hh_=rhh, A_cp_=acp,
                    B_hp_=bhp, D_pe_=dpe,
                    updater_below=ub,
                    C_iW=ciw,C_ib=cib,C_uW=cuw,C_ub=cub,
                    C_rW=crw,C_rb=crb,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)
        else:
            sequences = [state_below, updater_below, reseter_below]
            non_sequences = [c, p_from_c]
            fn = lambda x, g, r,  h, c1,pc,whh,ghh,rhh,acp,bhp,dpe,ub,ciw,cib,cuw,cub,crw,crb: self.step_fprop(x, h,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc,
                    W_hh_=whh, G_hh_=ghh, R_hh_=rhh, A_cp_=acp,
                    B_hp_=bhp, D_pe_=dpe,
                    updater_below=ub,
                    C_iW=ciw,C_ib=cib,C_uW=cuw,C_ub=cub,
                    C_rW=crw,C_rb=crb,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)

        non_sequences += non_seqs

        outputs_info = [init_state, None]
        if return_alignment:
            outputs_info.append(None)

        rval, updates = theano.scan(fn,
                        sequences=sequences,
                        non_sequences=non_sequences,
                        outputs_info=outputs_info,
                        name='layer_%s'%self.name,
                        truncate_gradient=truncate_gradient,
                        n_steps=nsteps,
                        strict=True)

        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out

class ReplicateLayer(Layer):

    def __init__(self, n_times):
        self.n_times = n_times
        super(ReplicateLayer, self).__init__(0, 0, None)

    def fprop(self, x):
        # This is black magic based on broadcasting,
        # that's why variable names don't make any sense.
        a = TT.shape_padleft(x)
        padding = [1] * x.ndim
        b = TT.alloc(numpy.float32(1), self.n_times, *padding)
        self.out = a * b
        return self.out

class PadLayer(Layer):

    def __init__(self, required):
        self.required = required
        Layer.__init__(self, 0, 0, None)

    def fprop(self, x):
        if_longer = x[:self.required]
        padding = ReplicateLayer(TT.max([1, self.required - x.shape[0]]))(x[-1]).out
        if_shorter = TT.concatenate([x, padding])
        diff = x.shape[0] - self.required
        self.out = ifelse(diff < 0, if_shorter, if_longer)
        return self.out

class ZeroLayer(Layer):

    def fprop(self, x):
        self.out = TT.zeros(x.shape)
        return self.out

def none_if_zero(x):
    if x == 0:
        return None
    return x

class Maxout(object):

    def __init__(self, maxout_part):
        self.maxout_part = maxout_part

    def __call__(self, x):
        shape = x.shape
        if x.ndim == 1:
            shape1 = TT.cast(shape[0] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape1, shape2])
            x = x.max(1)
        elif x.ndim == 2:
            shape1 = TT.cast(shape[1] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape1, shape2])
            x = x.max(2)
        else: # x.ndim == 3
            shape1 = TT.cast(shape[2] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape[1], shape1, shape2])
            x = x.max(3)
        return x

def prefix_lookup(state, p, s):
    if '%s_%s'%(p,s) in state:
        return state['%s_%s'%(p, s)]
    return state[s]

class EncoderDecoderBase(object):
    """
    TODO
    """
    def __init__(self):
        self.params = []

    def _create_embedding_layers(self, enc=False):
        logger.debug("_create_embedding_layers")
        if enc and self.state['encoder_stack'] < 1 and not self.state['use_hier_enc']:
            n_hids = [self.state['dim']]
        else:
            n_hids = [self.state['rank_n_approx']]

        self.approx_embedder = MultiLayer(self.rng,
            n_in=self.state['n_sym_source']
                if self.prefix.find("enc") >= 0
                else self.state['n_sym_target'],
            n_hids=n_hids,
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embdr'.format(self.prefix),
            learn_bias=False, # TODO: Set this to False
            **self.default_kwargs)
        self.params += self.approx_embedder.params
        # We have 3 embeddings for each word in each level,
        # the one used as input,
        # the one used to control resetting gate,
        # the one used to control update gate.
        self.input_embedders = [lambda x : 0] * self.num_levels
        self.reset_embedders = [lambda x : 0] * self.num_levels
        self.update_embedders = [lambda x : 0] * self.num_levels
        embedder_kwargs = dict(self.default_kwargs)
        embedder_kwargs.update(dict(
            n_in=n_hids[0],
            n_hids=[self.state['dim'] * self.state['dim_mult']],
            activation=['lambda x:x']))

        for level in range(self.num_levels):
            self.input_embedders[level] = MultiLayer(
                self.rng,
                name='{}_input_embdr_{}'.format(self.prefix, level),
                **embedder_kwargs)
            self.params += self.input_embedders[level].params
            if prefix_lookup(self.state, self.prefix, 'rec_gating'):
                self.update_embedders[level] = MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_update_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)
                self.params += self.update_embedders[level].params
            if prefix_lookup(self.state, self.prefix, 'rec_reseting'):
                self.reset_embedders[level] =  MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_reset_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)
                self.params += self.reset_embedders[level].params

    def _create_inter_level_layers(self):
        logger.debug("_create_inter_level_layers")
        inter_level_kwargs = dict(self.default_kwargs)
        inter_level_kwargs.update(
                n_in=self.state['dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'])

        self.inputers = [0] * self.num_levels
        self.reseters = [0] * self.num_levels
        self.updaters = [0] * self.num_levels

        for level in range(1, self.num_levels):
            self.inputers[level] = MultiLayer(self.rng,
                    name="{}_inputer_{}".format(self.prefix, level),
                    **inter_level_kwargs)
            self.params += self.inputers[level].params
            if prefix_lookup(self.state, self.prefix, 'rec_reseting'):
                self.reseters[level] = MultiLayer(self.rng,
                    name="{}_reseter_{}".format(self.prefix, level),
                    **inter_level_kwargs)
                self.params += self.reseters[level].params
            if prefix_lookup(self.state, self.prefix, 'rec_gating'):
                self.updaters[level] = MultiLayer(self.rng,
                    name="{}_updater_{}".format(self.prefix, level),
                    **inter_level_kwargs)
                self.params += self.updaters[level].params

    def _create_transition_layers(self):
        logger.debug("_create_transition_layers")
        self.transitions = []
        rec_layer = eval(prefix_lookup(self.state, self.prefix, 'rec_layer'))
        add_args = dict()

        if rec_layer == RecurrentLayerWithSearch:
            add_args = dict(c_dim=self.state['c_dim'])
            if self.state['deep_attention']:
                add_args['deep_attention'] = self.state['deep_attention']
                add_args['deep_attention_n_hids'] = self.state['deep_attention_n_hids']
                add_args['deep_attention_acts'] = copy.deepcopy(self.state['deep_attention_acts']) \
                                                    if isinstance(self.state['deep_attention_acts'], list) else \
                                                    self.state['deep_attention_acts']

        for level in range(self.num_levels):
            if level == 0:
                self.transitions.append(rec_layer(
                        self.rng,
                        n_hids=self.state['dim'],
                        activation=prefix_lookup(self.state, self.prefix, 'activ'),
                        bias_scale=self.state['bias'],
                        init_fn=(self.state['rec_weight_init_fn']
                            if not self.skip_init
                            else "sample_zeros"),
                        scale=prefix_lookup(self.state, self.prefix, 'rec_weight_scale'),
                        weight_noise=self.state['weight_noise_rec'],
                        dropout=self.state['dropout_rec'],
                        gating=prefix_lookup(self.state, self.prefix, 'rec_gating'),
                        gater_activation=prefix_lookup(self.state, self.prefix, 'rec_gater'),
                        reseting=prefix_lookup(self.state, self.prefix, 'rec_reseting'),
                        reseter_activation=prefix_lookup(self.state, self.prefix, 'rec_reseter'),
                        name='{}_transition_{}'.format(self.prefix, level),
                        **add_args))
                self.params += self.transitions[level].params
            else:
                # This is hardcoded to GRU for now
                add_args = dict()
                self.transitions.append(RecurrentLayer(
                        self.rng,
                        n_hids=self.state['dim'],
                        activation=prefix_lookup(self.state, self.prefix, 'activ'),
                        bias_scale=self.state['bias'],
                        init_fn=(self.state['rec_weight_init_fn']
                            if not self.skip_init
                            else "sample_zeros"),
                        scale=prefix_lookup(self.state, self.prefix, 'rec_weight_scale'),
                        weight_noise=self.state['weight_noise_rec'],
                        dropout=self.state['dropout_rec'],
                        gating=prefix_lookup(self.state, self.prefix, 'rec_gating'),
                        gater_activation=prefix_lookup(self.state, self.prefix, 'rec_gater'),
                        reseting=prefix_lookup(self.state, self.prefix, 'rec_reseting'),
                        reseter_activation=prefix_lookup(self.state, self.prefix, 'rec_reseter'),
                        name='{}_transition_{}'.format(self.prefix, level),
                        **add_args))
                self.params += self.transitions[level].params

class Encoder(EncoderDecoderBase):

    def __init__(self, state, rng, prefix='enc', skip_init=False):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init
        super(Encoder, self).__init__()

        self.num_levels = self.state['encoder_stack']

        if "use_hier_enc" in self.state and self.state['use_hier_enc']:
            logger.debug("Using bidirectional hierarchical encoder...")
            logger.debug("Changing state file for bidirectional hierarchical encoder.")
            self.state['forward'] = True
            self.state['backward'] = False
            self.state['last_forward'] = False
            self.state['last_backward'] = False

        # support multiple gating/memory units
        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.
        if 'hid_mult' not in self.state:
            self.state['hid_mult'] = 1.

    def create_layers(self):
        """
        Create all elements of Encoder's computation graph
        """

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers(enc=True)
        if self.num_levels > 0:
            self._create_transition_layers()
            self._create_inter_level_layers()
        self._create_representation_layers()

    def _create_representation_layers(self):
        logger.debug("_create_representation_layers")
        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        self.repr_contributors = [None] * self.num_levels
        for level in range(self.num_levels):
            self.repr_contributors[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=[self.state['dim'] * self.state['maxout_part']],
                activation=['lambda x: x'],
                name="{}_repr_contrib_{}".format(self.prefix, level),
                **self.default_kwargs)

            if not self.num_levels == 1 and not self.state['take_top']:
                self.params += self.repr_contributors[level].params

        self.repr_calculator = UnaryOp(
                activation=eval(self.state['unary_activ']),
                name="{}_repr_calc".format(self.prefix))
        self.params += self.repr_calculator.params

    def build_encoder(self, x,
            x_mask=None,
            use_noise=False,
            approx_embeddings=None,
            return_hidden_layers=False):
        """Create the computational graph of the RNN Encoder

        :param x:
            input variable, either vector of word indices or
            matrix of word indices, where each column is a sentence

        :param x_mask:
            when x is a matrix and input sequences are
            of variable length, this 1/0 matrix is used to specify
            the matrix positions where the input actually is

        :param use_noise:
            turns on addition of noise to weights
            (UNTESTED)

        :param approx_embeddings:
            forces encoder to use given embeddings instead of its own

        :param return_hidden_layers:
            if True, encoder returns all the activations of the hidden layer
            (WORKS ONLY IN NON-HIERARCHICAL CASE)
        """

        # Low rank embeddings of all the input words.
        # Shape in case of matrix input:
        #   (max_seq_len * batch_size, rank_n_approx),
        #   where max_seq_len is the maximum length of batch sequences.
        # Here and later n_words = max_seq_len * batch_size.
        # Shape in case of vector input:
        #   (seq_len, rank_n_approx)

        if not approx_embeddings:
            approx_embeddings = self.approx_embedder(x)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes: (n_words, dim)
        input_signals = []
        reset_signals = []
        update_signals = []

        for level in range(self.num_levels):
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

        # Hidden layers.
        # Shape in case of matrix input: (max_seq_len, batch_size, dim)
        # Shape in case of vector input: (seq_len, dim)

        hidden_layers = []
        betas = []
        if self.num_levels < 1:
            if x.ndim == 1:
                hidden_layers = [approx_embeddings.reshape([x.shape[0], self.state['dim']])]
            else:
                hidden_layers = [approx_embeddings.reshape([x.shape[0], x.shape[1],
                self.state['dim']])]
        else:
            for level in range(self.num_levels):
                # Each hidden layer (except the bottom one) receives
                # input, reset and update signals from below.
                # All the shapes: (n_words, dim)
                if level > 0:
                    input_signals[level] += self.inputers[level](hidden_layers[-1])
                    update_signals[level] += self.updaters[level](hidden_layers[-1])
                    reset_signals[level] += self.reseters[level](hidden_layers[-1])

                if self.state['use_hier_enc'] and \
                        'DoubleRecurrentLayer' == self.state['enc_rec_layer']:
                    result, beta = self.transitions[level](input_signals[level],
                                                  nsteps=x.shape[0],
                                                  batch_size=x.shape[1] if x.ndim == 2 else 1,
                                                  mask=x_mask,
                                                  gater_below=none_if_zero(update_signals[level]),
                                                  reseter_below=none_if_zero(reset_signals[level]),
                                                  use_noise=use_noise)
                    betas.append(beta)
                else:
                    result = self.transitions[level](input_signals[level],
                                                  nsteps=x.shape[0],
                                                  batch_size=x.shape[1] if x.ndim == 2 else 1,
                                                  mask=x_mask,
                                                  gater_below=none_if_zero(update_signals[level]),
                                                  reseter_below=none_if_zero(reset_signals[level]),
                                                  use_noise=use_noise)
                hidden_layers.append(result)
        if return_hidden_layers:
            assert self.state['encoder_stack'] <= 1
            if self.state['use_hier_enc'] and \
                    'DoubleRecurrentLayer' == self.state['enc_rec_layer']:
                return hidden_layers[0], betas[0]
            else:
                return hidden_layers[0]

        # If we no stack of RNN but only a usual one,
        # then the last hidden state is used as a representation.
        # Return value shape in case of matrix input:
        #   (batch_size, dim)
        # Return value shape in case of vector input:
        #   (dim,)
        if self.num_levels == 1 or self.state['take_top']:
            c = LastState()(hidden_layers[-1])
            if c.out.ndim == 2:
                c.out = c.out[:,:self.state['dim']]
            else:
                c.out = c.out[:self.state['dim']]
            return c
        else:
            # If we have a stack of RNN, then their last hidden states
            # are combined with a maxout layer.
            # Return value however has the same shape.
            contributions = []
            for level in range(self.num_levels):
                contributions.append(self.repr_contributors[level](
                    LastState()(hidden_layers[level])))
            # I do not know a good starting value for sum
            c = self.repr_calculator(sum(contributions[1:], contributions[0]))
            return c

class Decoder(EncoderDecoderBase):

    EVALUATION = 0
    SAMPLING = 1
    BEAM_SEARCH = 2

    def __init__(self, state,
                 rng, prefix='dec',
                 skip_init=False,
                 compute_alignment=False):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init
        self.compute_alignment = compute_alignment
        super(Decoder, self).__init__()
        if 'lm_readout_dim' not in self.state:
            if 'dim_lm' in state:
                self.state['lm_readout_dim'] = state['dim_lm']
            else:
                self.state['lm_readout_dim'] = state['dim']

        if 'controller_temp' not in self.state:
            self.state['controller_temp'] = 1.0

        if 'init_ctlr_bias' not in self.state:
            self.state['init_ctlr_bias'] = -1.0

        if 'rho' not in self.state:
            self.state['rho'] = 0.5

        if 'use_lm_control' not in self.state:
            self.state['use_lm_control'] = -1

        # should we really keep these separate?
        #if self.state['include_lm']:
            #self.state_lm = prototype_lm_state_en_finetune_union()
            #self.state_lm = prototype_lm_state_en_finetune()
            #self.state_lm = prototype_lm_state_en()

        # Actually there is a problem here -
        # we don't make difference between number of input layers
        # and outputs layers.
        self.num_levels = self.state['decoder_stack']

        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Decoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers()
        self._create_transition_layers()
        self._create_inter_level_layers()
        self._create_initialization_layers()
        self._create_decoding_layers()
        self._create_readout_layers()

        if self.state['include_lm']:
            self._create_lm()

        if self.state['search']:
            # TODO: check this when decoder_stack>1
            #assert self.num_levels == 1
            self.transitions[0].set_decoding_layers(
                    self.decode_inputers[0],
                    self.decode_reseters[0],
                    self.decode_updaters[0])


    def _create_lm(self):
        logger.debug("Creating External language model")

        if self.num_levels > 1:
            raise NotImplementedError('Fix this part!')
        level = 0
        self.lm_embedder = MultiLayer(
            self.rng,
            n_in=self.state['lm_readout_dim'],
            n_hids=self.state['dim'],
            activation=['lambda x:x'],
            learn_bias=False,
            name='{}_lm_embed_{}'.format(self.prefix, level),
            **self.default_kwargs)
        self.readout_params += self.lm_embedder.params

        if not self.state['use_arctic_lm']:
            self.LM_builder = LM_builder(self.state['lm_readout_dim'], self.rng, skip_init=False)
            # build output refers to the softmax over words
            self.LM_builder.__create_layers__(build_output=False)
            self.excluded_params = self.LM_builder.get_const_params()
        else:
            self.excluded_params = []

        # controller mlp for external language model,
        # generates a scaler btw [0,1] to weight language
        # model path, and conditioned on hidden_state_lm
        if self.state['use_lm_control'] == 1:
            act_str ='lambda x: TT.nnet.sigmoid(x / %f)' % self.state['controller_temp']
            self.lm_controller = MultiLayer(self.rng,
                                            n_in=self.state['lm_readout_dim'],
                                            n_hids=[1] if not
                                            self.state['vector_controller'] else
                                            [self.state['dim']],
                                            activation=[act_str],
                                            bias_scale=[self.state['init_ctlr_bias']],
                                            name='lm_controller')
            self.readout_params += self.lm_controller.params

        elif self.state['use_lm_control'] == 2:
            act_str ='lambda x: TT.nnet.sigmoid(x / %f)' % self.state['controller_temp']
            self.lm_controller = MultiLayer(self.rng,
                                            n_in=self.state['dim'],
                                            n_hids=[1] if not
                                            self.state['vector_controller']
                                            else [self.state['dim']],
                                            activation=[act_str],
                                            bias_scale=[self.state['init_ctlr_bias']],
                                            name='lm_controller')
            self.readout_params += self.lm_controller.params

        elif self.state['use_lm_control'] == 3:
            act_str ='lambda x: TT.nnet.sigmoid(x / %f)' % self.state['controller_temp']
            self.lm_controller_lm = MultiLayer(self.rng,
                                               n_in=self.state['lm_readout_dim'],
                                               n_hids=[1] if not
                                               self.state['vector_controller']
                                               else [self.state['dim']],
                                               activation=[act_str],
                                               bias_scale=[self.state['init_ctlr_bias']],
                                               name='lm_controller')
            self.lm_controller_tm = MultiLayer(self.rng,
                                               n_in=self.state['dim'],
                                               n_hids=[1] if not
                                               self.state['vector_controller']
                                               else [self.state['dim']],
                                               activation=[act_str],
                                               bias_scale=[self.state['init_ctlr_bias']],
                                               name='lm_controller')
            self.lm_controller = lambda x, y: TT.nnet.sigmoid(self.lm_controller_lm(x) + self.lm_controller_tm(y))
            self.readout_params += self.lm_controller.params

        act = lambda x: 1 - x
        self.prob_comp = UnaryOp(activation=act, name="prob_comp")
        self.one_layer = OneLayer()

        self.readout_params += self.prob_comp.params

    def _create_initialization_layers(self):
        logger.debug("_create_initialization_layers")
        self.initializers = [ZeroLayer()] * self.num_levels
        if self.state['bias_code']:
            for level in range(self.num_levels):
                self.initializers[level] = MultiLayer(self.rng,
                                                      n_in=self.state['dim'],
                                                      n_hids=[self.state['dim'] * self.state['hid_mult']],
                                                      activation=[prefix_lookup(self.state, 'dec', 'activ')],
                                                      bias_scale=[self.state['bias']],
                                                      name='{}_initializer_{}'.format(self.prefix, level),
                                                      **self.default_kwargs)
                self.params += self.initializers[level].params

    def _create_decoding_layers(self):
        logger.debug("_create_decoding_layers")
        self.decode_inputers = [lambda x : 0] * self.num_levels
        self.decode_reseters = [lambda x : 0] * self.num_levels
        self.decode_updaters = [lambda x : 0] * self.num_levels
        self.back_decode_inputers = [lambda x : 0] * self.num_levels
        self.back_decode_reseters = [lambda x : 0] * self.num_levels
        self.back_decode_updaters = [lambda x : 0] * self.num_levels

        decoding_kwargs = dict(self.default_kwargs)
        decoding_kwargs.update(dict(
                n_in=self.state['c_dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'],
                weight_noise=False,
                learn_bias=True))

        if self.state['decoding_inputs']:
            for level in range(self.num_levels):
                # Input contributions
                self.decode_inputers[level] = MultiLayer(
                    self.rng,
                    name='{}_dec_inputter_{}'.format(self.prefix, level),
                    **decoding_kwargs)
                self.params += self.decode_inputers[level].params
                # Update gate contributions
                if prefix_lookup(self.state, 'dec', 'rec_gating'):
                    self.decode_updaters[level] = MultiLayer(
                        self.rng,
                        name='{}_dec_updater_{}'.format(self.prefix, level),
                        **decoding_kwargs)
                    self.params += self.decode_updaters[level].params

                # Reset gate contributions
                if prefix_lookup(self.state, 'dec', 'rec_reseting'):
                    self.decode_reseters[level] = MultiLayer(
                        self.rng,
                        name='{}_dec_reseter_{}'.format(self.prefix, level),
                        **decoding_kwargs)
                    self.params += self.decode_reseters[level].params

    def _create_readout_layers(self):
        # created if we want to tune just the readout layer
        self.readout_params = []

        softmax_layer = self.state['softmax_layer'] if 'softmax_layer' in self.state \
                        else 'SoftmaxLayer'

        logger.debug("_create_readout_layers")

        readout_kwargs = dict(self.default_kwargs)
        readout_kwargs.update(dict(
                n_hids=self.state['dim'],
                activation='lambda x: x',
            ))

        self.repr_readout = MultiLayer(
                self.rng,
                n_in=self.state['c_dim'],
                learn_bias=True,
                name='{}_repr_readout'.format(self.prefix),
                **readout_kwargs)

        self.readout_params += self.repr_readout.params

        # Attention - this is the only readout layer
        # with trainable bias. Should be careful with that.
        self.hidden_readouts = [None] * self.num_levels
        for level in range(self.num_levels):
            self.hidden_readouts[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                name='{}_hid_readout_{}'.format(self.prefix, level),
                **readout_kwargs)
            self.readout_params += self.hidden_readouts[level].params

        self.prev_word_readout = 0
        if self.state['bigram']:
            self.prev_word_readout = MultiLayer(
                self.rng,
                n_in=self.state['rank_n_approx'],
                n_hids=self.state['dim'],
                activation=['lambda x:x'],
                learn_bias=False,
                name='{}_prev_readout_{}'.format(self.prefix, level),
                **self.default_kwargs)
            self.readout_params += self.prev_word_readout.params

        if self.state['deep_out']:
            act_layer = UnaryOp(activation=eval(self.state['unary_activ']))
            drop_layer = DropOp(rng=self.rng, dropout=self.state['dropout'])
            self.output_nonlinearities = [act_layer, drop_layer]
            self.output_layer = eval(softmax_layer)(
                    self.rng,
                    self.state['dim'] / self.state['maxout_part'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='{}_deep_softmax'.format(self.prefix),
                    use_nce=self.state['use_nce'] if 'use_nce' in self.state else False,
                    optimize_probs=self.state['optimize_probs'],
                    **self.default_kwargs)
            self.readout_params += act_layer.params
            self.readout_params += drop_layer.params
            self.readout_params += self.output_layer.params
        else:
            self.output_nonlinearities = []
            self.output_layer = eval(softmax_layer)(
                    self.rng,
                    self.state['dim'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='dec_softmax',
                    sum_over_time=True,
                    use_nce=self.state['use_nce'] if 'use_nce' in self.state else False,
                    optimize_probs=self.state['optimize_probs'],
                    **self.default_kwargs)
            self.readout_params += self.output_layer.params

    def build_decoder(self, c, y,
            c_mask=None,
            y_mask=None,
            step_num=None,
            mode=EVALUATION,
            given_init_states=None,
            T=1,
            prev_hid=None,
            prev_memory=None):
        """Create the computational graph of the RNN Decoder.

        :param c:
            representations produced by an encoder.
            (n_samples, dim) matrix if mode == sampling or
            (max_seq_len, batch_size, dim) matrix if mode == evaluation

        :param c_mask:
            if mode == evaluation a 0/1 matrix identifying valid positions in c

        :param y:
            if mode == evaluation
                target sequences, matrix of word indices of shape (max_seq_len, batch_size),
                where each column is a sequence
            if mode != evaluation
                a vector of previous words of shape (n_samples,)

        :param y_mask:
            if mode == evaluation a 0/1 matrix determining lengths
                of the target sequences, must be None otherwise

        :param mode:
            chooses on of three modes: evaluation, sampling and beam_search

        :param given_init_states:
            for sampling and beam_search. A list of hidden states
                matrices for each layer, each matrix is (n_samples, dim)

        :param T:
            sampling temperature
        """

        # Check parameter consistency
        if mode == Decoder.EVALUATION:
            assert not given_init_states
        else:
            assert not y_mask
            assert given_init_states
            if mode == Decoder.BEAM_SEARCH:
                assert T == 1

        # For log-likelihood evaluation the representation
        # be replicated for conveniency. In case backward RNN is used
        # it is not done.
        # Shape if mode == evaluation
        #   (max_seq_len, batch_size, dim)
        # Shape if mode != evaluation
        #   (n_samples, dim)
        if not self.state['search']:
            if mode == Decoder.EVALUATION:
                c = PadLayer(y.shape[0])(c)
            else:
                assert step_num
                c_pos = TT.minimum(step_num, c.shape[0] - 1)

        # Low rank embeddings of all the input words.
        # Shape if mode == evaluation
        #   (n_words, rank_n_approx),
        # Shape if mode != evaluation
        #   (n_samples, rank_n_approx)
        approx_embeddings = self.approx_embedder(y)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes if mode == evaluation:
        #   (n_words, dim)
        # where: n_words = max_seq_len * batch_size
        # All the shape if mode != evaluation:
        #   (n_samples, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            if level == 0:
                # Contributions directly from input words.
                input_signals.append(self.input_embedders[level](approx_embeddings))
                update_signals.append(self.update_embedders[level](approx_embeddings))
                reset_signals.append(self.reset_embedders[level](approx_embeddings))

                # Contributions from the encoded source sentence.
                if not self.state['search']:
                    current_c = c if mode == Decoder.EVALUATION else c[c_pos]
                    input_signals[level] += self.decode_inputers[level](current_c)
                    update_signals[level] += self.decode_updaters[level](current_c)
                    reset_signals[level] += self.decode_reseters[level](current_c)
            else:
                pass
                #raise NotImplementedError("should implement this!")

        # Hidden layers' initial states.
        # Shapes if mode == evaluation:
        #   (batch_size, dim)
        # Shape if mode != evaluation:
        #   (n_samples, dim)
        init_states = given_init_states
        if not init_states:
            init_states = []
            for level in range(self.num_levels):
                # TODO: check this if decoder_stack>1
                init_c = c[0, :, -self.state['dim']:]
                init_states.append(self.initializers[level](init_c))

        # Hidden layers' states.
        # Shapes if mode == evaluation:
        #  (seq_len, batch_size, dim)
        # Shapes if mode != evaluation:
        #  (n_samples, dim)
        hidden_layers = []
        contexts = []

        # Default value for alignment must be smth computable
        alignment = TT.zeros((1,))
        for level in range(self.num_levels):
            if level > 0:
                input_signals[level] += self.inputers[level](hidden_layers[level - 1])
                update_signals[level] += self.updaters[level](hidden_layers[level - 1])
                reset_signals[level] += self.reseters[level](hidden_layers[level - 1])

            add_kwargs = (dict(state_before=init_states[level])
                        if mode != Decoder.EVALUATION
                        else dict(init_state=init_states[level],
                            batch_size=y.shape[1] if y.ndim == 2 else 1,
                            nsteps=y.shape[0]))

            # TODO: fix this for decoder_stack>1
            if self.state['search'] and level == 0:
                add_kwargs['c'] = c
                add_kwargs['c_mask'] = c_mask
                add_kwargs['return_alignment'] = self.compute_alignment
                if mode != Decoder.EVALUATION:
                    add_kwargs['step_num'] = step_num

            result = self.transitions[level](input_signals[level],
                                             mask=y_mask,
                                             gater_below=none_if_zero(update_signals[level]),
                                             reseter_below=none_if_zero(reset_signals[level]),
                                             one_step=mode != Decoder.EVALUATION,
                                             use_noise=mode == Decoder.EVALUATION,
                                             **add_kwargs)

            if self.state['search']:
                if level == 0:
                    if self.compute_alignment:
                        #This implicitly wraps each element of result.out with a Layer to keep track of the parameters.
                        #It is equivalent to h=result[0], ctx=result[1] etc.
                        h, ctx, alignment = result
                        if mode == Decoder.EVALUATION:
                            alignment = alignment.out
                    else:
                        #This implicitly wraps each element of result.out with a Layer to keep track of the parameters.
                        #It is equivalent to h=result[0], ctx=result[1]
                        h, ctx = result
                else:
                    h = result
            # TODO:check this condition if decoder_stack>1
            else:
                h = result
                if mode == Decoder.EVALUATION:
                    ctx = c
                else:
                    ctx = ReplicateLayer(given_init_states[0].shape[0])(c[c_pos]).out

            hidden_layers.append(h)
            contexts.append(ctx)

        # In hidden_layers we do no have the initial state, but we need it.
        # Instead of it we have the last one, which we do not need.
        # So what we do is discard the last one and prepend the initial one.
        if mode == Decoder.EVALUATION:
            for level in range(self.num_levels):
                hidden_layers[level].out = TT.concatenate([
                    TT.shape_padleft(init_states[level].out),
                        hidden_layers[level].out])[:-1]

        # The output representation to be fed in softmax.
        # Shape if mode == evaluation
        #   (n_words, dim_r)
        # Shape if mode != evaluation
        #   (n_samples, dim_r)
        # ... where dim_r depends on 'deep_out' option.
        readout_tm = self.repr_readout(contexts[0])
        for level in range(self.num_levels):
            if mode != Decoder.EVALUATION:
                read_from = init_states[level]
            else:
                read_from = hidden_layers[level]
            read_from_var = read_from if type(read_from) == theano.tensor.TensorVariable else read_from.out
            if read_from_var.ndim == 3:
                read_from_var = read_from_var[:,:,:self.state['dim']]
            else:
                read_from_var = read_from_var[:,:self.state['dim']]
            if type(read_from) != theano.tensor.TensorVariable:
                read_from.out = read_from_var
            else:
                read_from = read_from_var
            readout_tm += self.hidden_readouts[level](read_from)

        if self.state['bigram']:
            if mode != Decoder.EVALUATION:
                check_first_word = (y > 0
                    if self.state['check_first_word']
                    else TT.ones((y.shape[0]), dtype="float32"))
                # padright is necessary as we want to multiply each row with a certain scalar
                readout_tm += TT.shape_padright(check_first_word) * self.prev_word_readout(approx_embeddings).out
            else:
                if y.ndim == 1:
                    readout_tm += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], 1, self.state['dim'])))
                else:
                    # This place needs explanation. When prev_word_readout is applied to
                    # approx_embeddings the resulting shape is
                    # (n_batches * sequence_length, repr_dimensionality). We first
                    # transform it into 3D tensor to shift forward in time. Then
                    # reshape it back.
                    readout_tm += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], y.shape[1], self.state['dim']))).reshape(
                                readout_tm.out.shape)

        # add language model, for evaluation lm_hidden_state is shifted forward in time
        # for one step and then multiplied by alpha. For sampling and search,
        # lm_hidden_state is zeroed if predicted word is 0 and then multiplied by alpha
        # This snippet is ugly
        # act = lambda x: -x + 1
        lm_alpha = []
        if self.state['include_lm']:
            if self.state['use_arctic_lm']:
                lstm_y = y
                lm_hidden_state, lm_memory = get_layer_graph(dim_word=self.state['rank_n_approx'],
                                                             dim=self.state['lm_readout_dim'],
                                                             n_words=self.state['null_sym_target'],
                                                             batch_size=self.state['bs'],
                                                             modelpath=self.state['modelpath'],
                                                             prev_state=prev_hid,
                                                             reload_=True,
                                                             prev_mem=prev_memory,
                                                             mode=mode,
                                                             inp=lstm_y)
            else:
                lm_hidden_state = self.LM_builder.build_for_translation(y, y_mask, prev_hid=prev_hid)

            check_first_word = (y > 0
                if self.state['check_first_word']
                else TT.ones((y.shape[0]), dtype="float32"))

            # Condition on lm only based controller
            if self.state['use_lm_control'] == 1:
                if mode != Decoder.EVALUATION:
                    lm_hidden_state = TT.switch(check_first_word[:,None],
                                                lm_hidden_state, 0. * lm_hidden_state)

                    lm_alpha = self.lm_controller(lm_hidden_state)
                    readout_lm = TT.shape_padright(check_first_word) * self.lm_embedder(lm_hidden_state).out
                    readout_lm = (lm_alpha.reshape((readout_lm.shape[0],
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim'])) *
                                  readout_lm).reshape(readout_tm.out.shape)
                    readout = readout_tm + readout_lm

                else:
                    lm_alpha = self.lm_controller(Shift()(lm_hidden_state))
                    if y.ndim == 1:
                        lm_alpha = lm_alpha.reshape((y.shape[0], 1,
                                                     1 if not
                                                     self.state['vector_controller']
                                                     else self.state['dim']))
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape(
                                             (y.shape[0], 1, self.state['dim']))
                        readout_lm = Shift()(readout_lm).reshape(readout_tm.out.shape)
                    else:
                        lm_alpha = lm_alpha.reshape((y.shape[0], y.shape[1],
                                                     1 if not
                                                     self.state['vector_controller']
                                                     else self.state['dim']))
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape((y.shape[0], y.shape[1], self.state['dim']))
                        readout_lm = Shift()(readout_lm)

                    scaled_readout_lm = (lm_alpha * readout_lm).reshape(readout_tm.out.shape)
                    readout = readout_tm + scaled_readout_lm

            # Condition on tm only based controller
            elif self.state['use_lm_control'] == 2:
                if mode != Decoder.EVALUATION:
                    lm_hidden_state = TT.switch(check_first_word[:,None],
                                                lm_hidden_state, 0. * lm_hidden_state)

                    lm_alpha = self.lm_controller(readout_tm)
                    readout_lm = TT.shape_padright(check_first_word) * self.lm_embedder(lm_hidden_state).out
                    readout_lm = (lm_alpha.reshape((readout_lm.shape[0],
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim'])) *
                                                    readout_lm).reshape(readout_tm.out.shape)
                    readout = readout_rm + readout_lm

                else:
                    lm_alpha = self.lm_controller(readout_tm)
                    if y.ndim == 1:
                        lm_alpha = lm_alpha.reshape((y.shape[0], 1,
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim']))
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape(
                                             (y.shape[0], 1, self.state['dim']))
                        readout_lm = Shift()(readout_lm).reshape(readout_tm.out.shape)
                    else:
                        lm_alpha = lm_alpha.reshape((y.shape[0], y.shape[1],
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim']))
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape((y.shape[0], y.shape[1], self.state['dim']))
                        readout_lm = Shift()(readout_lm)

                    readout = readout_tm + (lm_alpha * readout_lm).reshape(readout_tm.out.shape)

            # Conditiona on both tm and lm based controller
            elif self.state['use_lm_control'] == 3:
                if mode != Decoder.EVALUATION:
                    lm_hidden_state = TT.switch(check_first_word[:,None],
                                                lm_hidden_state, 0. * lm_hidden_state)

                    lm_alpha = self.lm_controller(readout_tm, lm_hidden_state)
                    readout_lm = TT.shape_padright(check_first_word) * self.lm_embedder(lm_hidden_state).out
                    readout_lm = (lm_alpha.reshape((readout_lm.shape[0],
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim'])) *
                                                    readout_lm).reshape(readout_tm.out.shape)
                    readout = readout_tm + readout_lm

                else:
                    lm_alpha = self.lm_controller(readout_tm, Shift()(lm_hidden_state))
                    if y.ndim == 1:
                        lm_alpha = lm_alpha.reshape((y.shape[0], 1,
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim']))
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape(
                                             (y.shape[0], 1, self.state['dim']))
                        readout_lm = Shift()(readout_lm).reshape(readout_tm.out.shape)
                    else:
                        lm_alpha = lm_alpha.reshape((y.shape[0], y.shape[1],
                                                    1 if not
                                                    self.state['vector_controller']
                                                    else self.state['dim']))
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape((y.shape[0], y.shape[1], self.state['dim']))
                        readout_lm = Shift()(readout_lm)

                    readout = readout_tm + (lm_alpha * readout_lm).reshape(readout.out.shape)

            # No lm controller, take convex combination by rho
            else:
                if mode != Decoder.EVALUATION:
                    lm_hidden_state = TT.switch(check_first_word[:,None],
                                                lm_hidden_state, 0. * lm_hidden_state)

                    readout_lm = TT.shape_padright(check_first_word) * self.lm_embedder(lm_hidden_state).out
                    readout_tm = readout_tm * (self.state['rho']) + readout_lm * (1 - self.state['rho'])
                else:
                    if y.ndim == 1:
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape(
                                             (y.shape[0], 1, self.state['dim']))
                        readout_lm = Shift()(readout_lm).reshape(readout_tm.out.shape)
                    else:
                        readout_lm = self.lm_embedder(lm_hidden_state).reshape((y.shape[0], y.shape[1], self.state['dim']))
                        readout_lm = Shift()(readout_lm).reshape(readout_tm.out.shape)

                    readout = readout_tm * self.state['rho'] + readout_lm * (1 - self.state['rho'])
            readout.params += readout_lm.params
            readout.params_grad_scale += readout_lm.params_grad_scale
        else:
            readout = readout_tm


        for fun in self.output_nonlinearities:
            if isinstance(fun, DropOp) and mode != Decoder.EVALUATION:
                readout = fun(readout, use_noise=False)
            else:
                readout = fun(readout)

        if mode == Decoder.SAMPLING:
            sample = self.output_layer.get_sample(
                    state_below=readout,
                    temp=T)
            # Current SoftmaxLayer.get_cost is stupid,
            # that's why we have to reshape a lot.
            self.output_layer.get_cost(state_below=readout.out,
                                       temp=T,
                                       target=sample)

            log_prob = self.output_layer.cost_per_sample
            retvals = [sample] + [log_prob] + hidden_layers
            if self.state['include_lm']:
                retvals += [lm_hidden_state]
                if self.state['use_arctic_lm']:
                    retvals += [lm_memory]
            return retvals
        elif mode == Decoder.BEAM_SEARCH:
            return self.output_layer(
                    state_below=readout.out,
                    temp=T).out
        elif mode == Decoder.EVALUATION:
            return (self.output_layer.train(
                    state_below=readout,
                    target=y,
                    mask=y_mask,
                    reg=None),
                    alignment, lm_alpha)
        else:
            raise Exception("Unknown mode for build_decoder")


    def sampling_step(self, *args):
        """Implements one step of sampling
            Args are necessary since the number (and the order) of arguments can vary
        """

        args = iter(args)

        # Arguments that correspond to scan's "sequences" parameteter:
        step_num = next(args)
        assert step_num.ndim == 0

        # Arguments that correspond to scan's "outputs" parameteter:
        prev_word = next(args)
        assert prev_word.ndim == 1
        # skip the previous word log probability
        assert next(args).ndim == 1
        prev_hidden_states = [next(args) for k in range(self.num_levels)]
        assert prev_hidden_states[0].ndim == 2

        if self.state['include_lm']:
            last_hid_state = next(args)
        else:
            last_hid_state = None

        if self.state['use_arctic_lm']:
            assert self.state['include_lm']
            last_memory = next(args)
        else:
            last_memory = None

        # Arguments that correspond to scan's "non_sequences":
        c = next(args)
        assert c.ndim == 2
        T = next(args)
        assert T.ndim == 0

        decoder_args = dict(given_init_states=prev_hidden_states, T=T, c=c,
                            prev_hid=last_hid_state, prev_memory=last_memory)

        sample, log_prob = self.build_decoder(y=prev_word, step_num=step_num, mode=Decoder.SAMPLING, **decoder_args)[:2]
        hidden_states = self.build_decoder(y=sample, step_num=step_num, mode=Decoder.SAMPLING, **decoder_args)[2:]

        return [sample, log_prob] + hidden_states

    def build_initializers(self, c):
        return [init(c).out for init in self.initializers]

    def build_sampler(self, n_samples, n_steps, T, c):

        states = [TT.zeros(shape=(n_samples,), dtype='int64'),
                   TT.zeros(shape=(n_samples,), dtype='float32')]

        init_c = c[0, -self.state['dim']:]
        states += [ReplicateLayer(n_samples)(init(init_c).out).out for init in self.initializers]

        # we need to add another list of zeros for the initial state of the hiddens
        if self.state['include_lm']:
            states += [TT.zeros(shape=(n_samples, self.state['lm_readout_dim']), dtype='float32')]
            #Add the memory intializer for the lstm.
            if self.state['use_arctic_lm']:
                states += [TT.zeros(shape=(n_samples, self.state['lm_readout_dim']), dtype='float32')]

        if not self.state['search']:
            c = PadLayer(n_steps)(c).out

        # Pad with final states
        non_sequences = [c, T]

        outputs, updates = theano.scan(self.sampling_step,
                                       outputs_info=states,
                                       non_sequences=non_sequences,
                                       sequences=[TT.arange(n_steps, dtype="int64")],
                                       n_steps=n_steps,
                                       name="{}_sampler_scan".format(self.prefix))

        return (outputs[0], outputs[1]), updates

    def build_next_probs_predictor(self, c, step_num, y, init_states, prev_hid, prev_memory=None):
        return self.build_decoder(c, y, mode=Decoder.BEAM_SEARCH,
                                  given_init_states=init_states,
                                  step_num=step_num,
                                  prev_hid=prev_hid,
                                  prev_memory=prev_memory)

    def build_next_states_computer(self, c, step_num, y, init_states, prev_hid, prev_memory=None):
        return self.build_decoder(c, y, mode=Decoder.SAMPLING,
                                  given_init_states=init_states,
                                  step_num=step_num,
                                  prev_hid=prev_hid,
                                  prev_memory=prev_memory)[2:]

class RNNEncoderDecoder(object):
    """This class encapsulates the translation model.

    The expected usage pattern is:
    >>> encdec = RNNEncoderDecoder(...)
    >>> encdec.build(...)
    >>> useful_smth = encdec.create_useful_smth(...)

    Functions from the create_smth family (except create_lm_model)
    when called complile and return functions that do useful stuff.
    """

    def __init__(self, state, rng,
                 skip_init=False,
                 compute_alignment=False):
        """Constructor.

        :param state:
            A state in the usual groundhog sense.
        :param rng:
            Random number generator. Something like numpy.random.RandomState(seed).
        :param skip_init:
            If True, all the layers are initialized with zeros. Saves time spent on
            parameter initialization if they are loaded later anyway.
        :param compute_alignment:
            If True, the alignment is returned by the decoder.
        """
        self.params = []
        self.state = state
        self.rng = rng
        self.skip_init = skip_init
        self.compute_alignment = compute_alignment

        if "use_hier_enc" in self.state and self.state['use_hier_enc']:
            logger.debug("Using bidirectional hierarchical encoder...")
            logger.debug("Changing state file for bidirectional hierarchical encoder.")
            self.state['forward'] = True
            self.state['backward'] = False
            self.state['last_forward'] = False
            self.state['last_backward'] = False


        if 'include_lm' not in self.state:
            self.state['include_lm'] = False
        if 'use_arctic_lm' not in self.state:
            self.state['use_arctic_lm'] = False
        if 'shallow_lm' not in self.state:
            self.state['shallow_lm'] = False
        if 'train_only_readout' not in self.state:
            self.state['train_only_readout'] = False
        if 'mask_first_lm' not in self.state:
            self.state['mask_first_lm'] = False
        if 'random_readout' not in self.state:
            self.state['random_readout'] = False
        if 'source_splitted' not in self.state:
            self.state['source_splitted'] = False
        if 'optimize_probs' not in self.state:
            self.state['optimize_probs'] = True
        if 'use_hier_enc' not in self.state:
            self.state['use_hier_enc'] = False
        if 'vector_controller' not in self.state:
            self.state['vector_controller'] = False
        if 'additional_ngrad_monitors' not in self.state:
            self.state['additional_ngrad_monitors'] = None

    def build(self, use_noise=True):
        logger.debug("Create input variables")
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')
        self.inputs = [self.x, self.y, self.x_mask, self.y_mask]

        # Annotation for the log-likelihood computation
        training_c_components = []

        logger.debug("Create encoder")
        self.encoder = Encoder(self.state, self.rng,
                prefix="enc",
                skip_init=self.skip_init)
        self.encoder.create_layers()

        logger.debug("Build encoding computation graph")
        if self.state['use_hier_enc'] and \
            self.state['enc_rec_layer'] == 'DoubleRecurrentLayer':
            forward_training_c, self.betas = self.encoder.build_encoder(
                        self.x, self.x_mask,
                        use_noise=self.state['use_noise'],
                        return_hidden_layers=True)
        else:
            forward_training_c = self.encoder.build_encoder(
                    self.x, self.x_mask,
                    use_noise=self.state['use_noise'],
                    return_hidden_layers=True)

        if self.state['encoder_stack'] > 0 and not self.state['use_hier_enc']:
            logger.debug("Create backward encoder")
            self.backward_encoder = Encoder(self.state, self.rng,
                    prefix="back_enc",
                    skip_init=self.skip_init)
            self.backward_encoder.create_layers()

            logger.debug("Build backward encoding computation graph")
            backward_training_c = self.backward_encoder.build_encoder(
                    self.x[::-1],
                    self.x_mask[::-1],
                    use_noise=self.state['use_noise'],
                    approx_embeddings=self.encoder.approx_embedder(self.x[::-1]),
                    return_hidden_layers=True)

            # Reverse time for backward representations.
            backward_training_c.out = backward_training_c.out[::-1]

        if self.state['encoder_stack'] < 1 or self.state['use_hier_enc']:
            training_c_components.append(forward_training_c)
        else:
            if self.state['forward']:
                training_c_components.append(forward_training_c)

            if self.state['last_forward']:
                training_c_components.append(
                        ReplicateLayer(self.x.shape[0])(forward_training_c[-1]))

            if self.state['backward']:
                training_c_components.append(backward_training_c)

            if self.state['last_backward']:
                training_c_components.append(ReplicateLayer(self.x.shape[0])
                                            (backward_training_c[0]))

        self.state['c_dim'] = len(training_c_components) * self.state['dim']
        logger.debug("Create decoder")
        self.decoder = Decoder(self.state, self.rng,
                               skip_init=self.skip_init,
                               compute_alignment=self.compute_alignment)

        self.decoder.create_layers()
        logger.debug("Build log-likelihood computation graph")

        self.predictions, self.alignment, self.lm_alpha = self.decoder.build_decoder(c=Concatenate(axis=2)(*training_c_components),
                                                                      c_mask=self.x_mask,
                                                                      y=self.y, y_mask=self.y_mask)

        # Annotation for sampling
        sampling_c_components = []
        self.params += self.encoder.params
        if self.state['encoder_stack'] > 0 and not self.state['use_hier_enc']:
            self.params += self.backward_encoder.params

        self.params += self.decoder.params
        self.params += self.decoder.readout_params

        #Remove the backward Encoder's approximate embedder,
        #because we don't use it!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        i = 0
        for param in  self.backward_encoder.params:
            if "approx_emb" in param.name:
                self.params.pop(i)
            i += 1
        '''

        logger.debug("Build sampling computation graph")
        self.sampling_x = TT.lvector("sampling_x")
        self.n_samples = TT.lscalar("n_samples")
        self.n_steps = TT.lscalar("n_steps")

        #softmax temperature
        self.T = TT.scalar("T")

        if self.state['use_hier_enc'] and \
            self.state['enc_rec_layer'] == 'DoubleRecurrentLayer':
            self.forward_sampling_c,\
            self.sampling_betas = self.encoder.build_encoder(self.sampling_x,
                                                             return_hidden_layers=True).out
        else:
            self.forward_sampling_c = self.encoder.build_encoder(self.sampling_x,
                                                                 return_hidden_layers=True).out

        if self.state['encoder_stack'] > 0 and not self.state['use_hier_enc']:
            self.backward_sampling_c = self.backward_encoder.build_encoder(
                    self.sampling_x[::-1],
                    approx_embeddings=self.encoder.approx_embedder(self.sampling_x[::-1]),
                    return_hidden_layers=True).out[::-1]
        if self.state['encoder_stack'] < 1 or self.state['use_hier_enc']:
            sampling_c_components.append(self.forward_sampling_c)
        else:
            if self.state['forward']:
                sampling_c_components.append(self.forward_sampling_c)

            if self.state['last_forward']:
                sampling_c_components.append(ReplicateLayer(self.sampling_x.shape[0])
                        (self.forward_sampling_c[-1]))
            if self.state['backward']:
                sampling_c_components.append(self.backward_sampling_c)

            if self.state['last_backward']:
                sampling_c_components.append(ReplicateLayer(self.sampling_x.shape[0])
                                            (self.backward_sampling_c[0]))

        self.sampling_c = Concatenate(axis=1)(*sampling_c_components).out

        (self.sample, self.sample_log_prob), self.sampling_updates =\
            self.decoder.build_sampler(self.n_samples, self.n_steps, self.T,
                                       c=self.sampling_c)

        logger.debug("Create auxiliary variables")
        self.c = TT.matrix("c")
        self.step_num = TT.lscalar("step_num")

        self.current_states = [TT.matrix("cur_{}".format(i))
                                         for i in range(self.decoder.num_levels)]

        self.gen_y = TT.lvector("gen_y")
        self.current_states_lm = TT.matrix("cur_lm")
        self.current_memory_lm = TT.matrix("cur_mem_lm")
        self.lst_state_lm = []
        self.lst_state_lm += [self.current_states_lm]
        self.lst_state_lm += [self.current_memory_lm]
        """
        if self.state['include_lm']:
           self.lst_state_lm = [self.current_states_lm]
           if self.state['use_arctic_lm']:
                self.lst_state_lm += [self.current_memory_lm]
        """


    def create_lm_model(self):
        # singleton constructor
        if hasattr(self, 'lm_model'):
            return self.lm_model

        # if we include a pre-trained lm, we should exclude its recurrent
        # weights and optionally its embedding matrix
        excluded_params = None
        if hasattr(self.decoder, 'excluded_params'):

            if self.state['train_only_readout']:
                # exclude the union of the NON-readout parameters and the whole graph (this
                # accounts for lm params) + the embeddings (leave these)
                # self.prediction.params are all the parameter
                excluded_params = list((set(self.predictions.params) - set(self.decoder.readout_params))\
                        | set(self.decoder.approx_embedder.params))
            # just exclude the language model
            else:
                excluded_params = self.decoder.excluded_params

            # verbose for debugging purposes
            if 'additional_excludes' in self.state and\
                self.state['additional_excludes']:
                excluded_params += [x for x in self.predictions.params\
                        if x.name in self.state['additional_excludes']]

            logger.debug("Excluding these params for training:")
            print excluded_params

            logger.debug("Training these params:")
            print list(set(self.predictions.params) - set(excluded_params))

        self.lm_model = LM_Model(cost_layer=self.predictions,
                                 sample_fn=self.create_sampler(),
                                 weight_noise_amount=self.state['weight_noise_amount'],
                                 indx_word=self.state['indx_word_target'],
                                 indx_word_src=self.state['indx_word'],
                                 rng=self.rng,
                                 exclude_params=excluded_params,
                                 additional_ngrad_monitors=self.state['additional_ngrad_monitors'])
        self.lm_model.lm_alpha = self.lm_alpha
        self.lm_model.name2pos = name2pos(self.lm_model.params)
        self.lm_model.load_dict(self.state)
        logger.debug("Model params:\n{}".format(
            pprint.pformat(sorted([p.name for p in self.lm_model.params]))))

        return self.lm_model

    def create_representation_computer(self):
        if not hasattr(self, "repr_fn"):
            self.repr_fn = theano.function(
                    inputs=[self.sampling_x],
                    outputs=[self.sampling_c],
                    name="repr_fn")
        return self.repr_fn

    def create_initializers(self):
        if not hasattr(self, "init_fn"):
            init_c = self.sampling_c[0, -self.state['dim']:]
            self.init_fn = theano.function(
                    inputs=[self.sampling_c],
                    outputs=self.decoder.build_initializers(init_c),
                    name="init_fn")
        return self.init_fn

    def create_sampler(self, many_samples=False):
        if hasattr(self, 'sample_fn'):
            return self.sample_fn
        logger.debug("Compile sampler")
        self.sample_fn = theano.function(
                inputs=[self.n_samples, self.n_steps, self.T, self.sampling_x],
                outputs=[self.sample, self.sample_log_prob],
                updates=self.sampling_updates,
                name="sample_fn")
        if not many_samples:
            def sampler(*args):
                return map(lambda x : x.squeeze(), self.sample_fn(1, *args))
            return sampler
        return self.sample_fn

    def create_scorer(self, batch=False):
        if not hasattr(self, 'score_fn'):
            logger.debug("Compile scorer")
            self.score_fn = theano.function(inputs=self.inputs,
                                            outputs=[-self.predictions.cost_per_sample],
                                            name="score_fn")

        if batch:
            return self.score_fn

        def scorer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            return self.score_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])

        return scorer

    def create_next_probs_computer(self):
        if not hasattr(self, 'next_probs_fn'):
            self.next_probs_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.lst_state_lm + self.current_states,
                    outputs=[self.decoder.build_next_probs_predictor(self.c, self.step_num, self.gen_y,
                                                                     self.current_states,
                                                                     self.current_states_lm,
                                                                     self.current_memory_lm)],
                    name="next_probs_fn",on_unused_input='warn')#,mode='DebugMode')
        return self.next_probs_fn

    def create_next_states_computer(self):
        if not hasattr(self, 'next_states_fn'):
            self.next_states_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.lst_state_lm + self.current_states,
                    outputs=self.decoder.build_next_states_computer(self.c, self.step_num, self.gen_y,
                                                                    self.current_states,
                                                                    self.current_states_lm,
                                                                    self.current_memory_lm),
                    name="next_states_fn",on_unused_input='warn')
        return self.next_states_fn

    def create_probs_computer(self, return_alignment=False):
        if not hasattr(self, 'probs_fn'):
            logger.debug("Compile probs computer")
            self.probs_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[self.predictions.word_probs, self.alignment],
                    name="probs_fn")
        def probs_computer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            probs, alignment = self.probs_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
            if return_alignment:
                return probs, alignment
            else:
                return probs
        return probs_computer

def parse_input(state, word2idx, line, raise_unk=False, idx2word=None, unk_sym=-1, null_sym=-1):
    if unk_sym < 0:
        unk_sym = state['unk_sym_source']
    if null_sym < 0:
        null_sym = state['null_sym_source']

    # for openmt15 chinese source side, chars splitted beforehand
    if state['source_encoding'] == 'utf8' and state['source_splitted'] is False:
        seqin = [l for l in line]
    else:
        seqin = line.split()

    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_sym)
        if seq[idx] >= state['n_sym_source']:
            seq[idx] = unk_sym
        if seq[idx] == unk_sym and raise_unk:
            raise Exception("Unknown word {}".format(sx))

    seq[-1] = null_sym
    if idx2word:
        idx2word[null_sym] = '<eos>'
        idx2word[unk_sym] = state['oov']
        parsed_in = [idx2word[sx] for sx in seq]
        return seq, " ".join(parsed_in)

    return seq, seqin
