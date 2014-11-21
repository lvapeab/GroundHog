import numpy
import logging
import pprint
import operator
import itertools
import copy
import theano
import cPickle
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate,\
        ReplicateLayer, \
        PadLayer, \
        ZeroLayer
from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator
from groundhog.utils import sample_zeros, sample_weights_orth, init_bias, sample_weights_classic
import groundhog.utils as utils
from experiments.nmt.language_model import LM_builder, LM_wrapper
import theano.printing
logger = logging.getLogger(__name__)

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
        self._print_shape = theano.printing.Print('!!! : Shape is : ',['shape'])

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
                   mask=None,
                   c=None,
                   c_mask=None,
                   p_from_c=None,
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

        W_hh = self.W_hh
        G_hh = self.G_hh
        R_hh = self.R_hh
        A_cp = self.A_cp
        B_hp = self.B_hp
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
        h = updater * h + (1-updater) * state_before

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before

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

        if mask:
            sequences = [state_below, mask, updater_below, reseter_below]
            non_sequences = [c, c_mask, p_from_c]
            #              seqs    | out |  non_seqs
            fn = lambda x, m, g, r,   h,   c1, cm, pc : self.step_fprop(x, h, mask=m,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc, c_mask=cm,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)
        else:
            sequences = [state_below, updater_below, reseter_below]
            non_sequences = [c, p_from_c]
            #            seqs   | out | non_seqs
            fn = lambda x, g, r,   h,    c1, pc : self.step_fprop(x, h,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)

        outputs_info = [init_state, None]
        if return_alignment:
            outputs_info.append(None)

        rval, updates = theano.scan(fn,
                        sequences=sequences,
                        non_sequences=non_sequences,
                        outputs_info=outputs_info,
                        name='layer_%s'%self.name,
                        truncate_gradient=truncate_gradient,
                        n_steps=nsteps)
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out

class RecurrentLayerWithSearchAndLM(RecurrentLayerWithSearch):
    """Inherits from RecurrentLayerWithSearch class"""

    def __init__(self,rng,
                 external_lm=None,
                 *args, 
                 **kwargs):
        logger.debug("RecurrentLayerWithSearchAndLM is used")

        # check if external lm parameter dict is in appropriate format
        assert type(external_lm) is dict and all([el in external_lm
                                                  for el in {'lm_state_file',
                                                             'lm_model_file',
                                                             'lm_type'}])

        # initialize actual recurrent layer with search
        super(RecurrentLayerWithSearchAndLM, self).__init__(rng,
                                                            *args,
                                                            **kwargs)

        # this initializes language model and loads model from file
        self.lm_wrapper = LM_wrapper(lm_type=external_lm['lm_type'],
                                     state_file=external_lm['lm_state_file'],
                                     model_file=external_lm['lm_model_file'],
                                     rng=rng)
        
        # merge parameters with recurrent layer parameter list
        self.merge_params(self.lm_wrapper)
        
        # exclude output layer parameters since they are not part of 
        # the computational graph (for now at least)
        #TODO: make it generic to handle output layer params
        for el in self.lm_wrapper.lm.output_layer.params:
            idx = self.params.index(el)
            del self.params[idx]
            del self.params_grad_scale[idx]

    def step_fprop(self,
                   state_below,
                   state_before,
                   lm_state_before,
                   gater_below=None,
                   reseter_below=None,
                   y=None,
                   mask=None,
                   c=None,
                   c_mask=None,
                   p_from_c=None,
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

        :type y: theano variable
        :param y:
            TODO: fix comment below
            if mode == evaluation
                target sequences, matrix of word indices of shape (max_seq_len, batch_size),
                where each column is a sequence
            if mode != evaluation
                a vector of previous words of shape (n_samples,)
        """

        updater_below = gater_below

        W_hh = self.W_hh
        G_hh = self.G_hh
        R_hh = self.R_hh
        A_cp = self.A_cp
        B_hp = self.B_hp
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
        ht = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        updater = self.updater_activation(TT.dot(state_before, G_hh) +
                updater_below)
        ht = updater * ht + (1-updater) * state_before

        if mask is not None:
            if ht.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            ht = mask * ht + (1-mask) * state_before

        # Feed previous label to the language model and
        # obtain its hidden representation
        next_lm = self.lm_wrapper.ht_sampler()
        hl = next_lm(y,lm_state_before)

        results = [ht, hl, ctx]
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
              return_alignment=False,
              y=None):

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

        if not isinstance(batch_size, int) or batch_size != 1:
            init_state_lm = TT.alloc(floatX(0), batch_size, self.lm_wrapper.n_hids)
        else:
            init_state_lm = TT.alloc(floatX(0), self.lm_wrapper.n_hids)

        p_from_c =  utils.dot(c, self.A_cp).reshape(
                (c.shape[0], c.shape[1], self.n_hids))

        if mask:
            sequences = [state_below, mask, updater_below, reseter_below, y]
            non_sequences = [c, c_mask, p_from_c]
            #              seqs        |   out   |  non_seqs
            fn = lambda x, m, g, r, y,   h, h_lm,  c1, cm, pc : self.step_fprop(x, h, h_lm, mask=m,
                    gater_below=g, reseter_below=r, y=y,
                    c=c1, p_from_c=pc, c_mask=cm,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)
        else:
            sequences = [state_below, updater_below, reseter_below, y]
            non_sequences = [c, p_from_c]
            #              seqs    | out | non_seqs
            fn = lambda x, g, r, y,   h,    c1, pc : self.step_fprop(x, h,
                    gater_below=g, reseter_below=r, y=y,
                    c=c1, p_from_c=pc,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)

        outputs_info = [init_state, init_state_lm, None]
        if return_alignment:
            outputs_info.append(None)

        rval, updates = theano.scan(fn,
                        sequences=sequences,
                        non_sequences=non_sequences,
                        outputs_info=outputs_info,
                        name='layer_%s'%self.name,
                        truncate_gradient=truncate_gradient,
                        n_steps=nsteps)
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out

