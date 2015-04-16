"""
Recurrent layers.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import copy
import theano
import theano.tensor as TT
# Nicer interface of scan
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog import utils
from groundhog.utils import sample_weights, \
        sample_weights_classic,\
        sample_weights_orth, \
        init_bias, \
        constant_shape, \
        sample_zeros
from basic import Layer

class RecurrentMultiLayer(Layer):
    """
    Constructs a recurrent layer whose transition from h_tm1 to h_t is given
    by an MLP or logistic regression. In our ICLR submission this is a
    DT-RNN model.
    """
    def __init__(self,
                 rng,
                 n_hids=[500,500],
                 activation = [TT.tanh, TT.tanh],
                 scale=.01,
                 sparsity = -1,
                 activ_noise=0.,
                 weight_noise=False,
                 dropout = 1.,
                 init_fn='sample_weights',
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 grad_scale = 1.,
                 profile = 0,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: list of ints
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        self.grad_scale = grad_scale
        if type(n_hids) not in (list, tuple):
            n_hids = [n_hids]
        n_layers = len(n_hids)
        if type(scale) not in (list, tuple):
            scale = [scale] * n_layers
        if type(sparsity) not in (list, tuple):
            sparsity = [sparsity] * n_layers
        for idx, sp in enumerate(sparsity):
            if sp < 0: sparsity[idx] = n_hids[idx]
        if type(activation) not in (list, tuple):
            activation = [activation] * n_layers
        if type(bias_scale) not in (list, tuple):
            bias_scale = [bias_scale] * (n_layers-1)
        if type(bias_fn) not in (list, tuple):
            bias_fn = [bias_fn] * (n_layers-1)
        if type(init_fn) not in (list, tuple):
            init_fn = [init_fn] * n_layers

        for dx in xrange(n_layers):
            if dx < n_layers-1:
                if type(bias_fn[dx]) is str or type(bias_fn[dx]) is unicode:
                    bias_fn[dx] = eval(bias_fn[dx])
            if type(init_fn[dx]) is str or type(init_fn[dx]) is unicode:
                init_fn[dx] = eval(init_fn[dx])
            if type(activation[dx]) is str or type(activation[dx]) is unicode:
                activation[dx] = eval(activation[dx])
        self.scale = scale
        self.n_layers = n_layers
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        assert rng is not None, "random number generator should not be empty!"
        super(RecurrentMultiLayer, self).__init__(n_hids[0],
                                                 n_hids[-1],
                                                 rng,
                                                 name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                        self.n_hids[dx],
                                        self.sparsity[dx],
                                        self.scale[dx],
                                        rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))
            if dx > 0:
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx-1](self.n_hids[dx],
                                       self.bias_scale[dx-1],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] + [x for x in self.b_hhs]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]


    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   init_state=None,
                   use_noise=True,
                   no_noise_bias=False):
        """
        Constructs the computational graph of a single step of the recurrent
        layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs, self.nW_hhs)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs, self.nb_hss)]
            else:
                b_hhs = self.b_hhs
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
        preactiv = TT.dot(state_before, W_hhs[0]) +state_below
        h = self.activation[0](preactiv)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval +=[h]
        for dx in xrange(1, self.n_layers):
            preactiv = TT.dot(h, W_hhs[dx]) + b_hhs[dx-1]
            h = self.activation[dx](preactiv)

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              n_steps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False):
        """
        Evaluates the forward through a recurrent layer

        :type state_below: theano variable
        :param state_below: the input of the recurrent layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type init_state: theano variable or None
        :param init_state: initial state for the hidden layer

        :type n_steps: None or int or theano scalar
        :param n_steps: Number of steps the recurrent netowrk does

        :type batch_size: int
        :param batch_size: the size of the minibatch over which scan runs

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type truncate_gradient: int
        :param truncate_gradient: If negative, no truncation is used,
            otherwise truncated BPTT is used, where you go backwards only this
            amount of steps

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """


        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if n_steps is None:
            n_steps = state_below.shape[0]
            if batch_size and batch_size != 1:
                n_steps = n_steps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((n_steps, batch_size, self.n_in))


        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids[0])
            else:
                init_state = TT.alloc(floatX(0), self.n_hids[0])

        if mask:
            inps = [state_below, mask]
            fn = lambda x,y,z : self.step_fprop(x,y,None, z, use_noise=use_noise,
                                               no_noise_bias=no_noise_bias)
        else:
            inps = [state_below]
            fn = lambda tx, ty: self.step_fprop(tx, None, None, ty,
                                                use_noise=use_noise,
                                                no_noise_bias=no_noise_bias)

        if self.dropout < 1. and use_noise:
            # build dropout mask outside scan
            allhid = numpy.sum(self.n_hids)
            shape = state_below.shape
            if state_below.ndim == 3:
                alldpmask = self.trng.binomial(
                        (n_steps, batch_size, allhid),
                        n = 1, p = self.dropout, dtype=state_below.dtype)
            else:
                alldpmask = self.trng.binomial(
                        (n_steps, allhid),
                        n = 1, p = self.dropout, dtype=state_below.dtype)
            inps.append(alldpmask)
            if mask:
                fn = lambda x,y,z,u : self.step_fprop(x,y,z,u,use_noise=use_noise)
            else:
                fn = lambda tx, ty, tu: self.step_fprop(tx,None,ty,tu,
                                                use_noise=use_noise)

        rval, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = [None]*(self.n_layers-1) +
                                    [init_state],
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        n_steps = n_steps)
        if not isinstance(rval,(list, tuple)):
            rval = [rval]
        new_h = rval[-1]
        self.out = rval[-1]
        self.rval = rval
        self.updates =updates

        return self.out


class RecurrentMultiLayerInp(RecurrentMultiLayer):
    """
    Similar to the RecurrentMultiLayer, with the exception that the input is
    fed into the top layer of the MLP (rather than being an input to the
    MLP).
    """
    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                        self.n_hids[dx],
                                        self.sparsity[dx],
                                        self.scale[dx],
                                        rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))
            if dx < self.n_layers-1:
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx](self.n_hids[dx],
                                       self.bias_scale[dx],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] + [x for x in self.b_hhs]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]


    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs,self.nW_hss)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs,self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs

        h = self.activation[0](TT.dot(state_before,
                                      W_hhs[0])+b_hhs[0])
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)

        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval += [h]
        for dx in xrange(1, self.n_layers-1):
            h = self.activation[dx](TT.dot(h,
                                           W_hhs[dx])+b_hhs[dx])
            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]
        h = self.activation[-1](TT.dot(h, W_hhs[-1]) + state_below)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                    dpidx = dpidx + h.shape[1]
                else:
                    h = h * dpmask[dpidx:dpidx+h.shape[0]]
                    dpidx = dpidx + h.shape[0]
            else:
                h = h * self.dropout
        rval += [h]
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

class RecurrentMultiLayerShortPath(RecurrentMultiLayer):
    """
    A similar layer to RecurrentMultiLayer (the DT-RNN), with the difference
    that we have shortcut connections in the MLP representing the transition
    from previous hidden state to the next
    """
    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        self.W_shortp = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                    self.n_hids[dx],
                                    self.sparsity[dx],
                                    self.scale[dx],
                                    rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))

            if dx > 0:
                W_shp = self.init_fn[dx](self.n_hids[self.n_layers-1],
                                         self.n_hids[dx],
                                         self.sparsity[dx],
                                         self.scale[dx],
                                         rng=self.rng)
                self.W_shortp.append(theano.shared(value=W_shp,
                                                   name='W_s%d_%s'%(dx,self.name)))
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx-1](self.n_hids[dx],
                                       self.bias_scale[dx-1],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] + [x for x in self.b_hhs] +\
                [x for x in self.W_shortp]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.nW_shortp = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_shortp]

            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs] + [x for x in self.nW_shortp]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]



    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs,self.nW_hhs)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs,self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
            W_shp = [(x+y) for x, y in zip(self.W_shortp,self.nW_shortp)]
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
            W_shp = self.W_shortp
        h = self.activation[0](TT.dot(state_before,
                                      W_hhs[0])+state_below)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout
        rval += [h]
        for dx in xrange(1, self.n_layers):
            h = self.activation[dx](TT.dot(h,
                                           W_hhs[dx])+
                                    TT.dot(state_before,
                                           W_shp[dx-1])+b_hhs[dx-1])
            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

class RecurrentMultiLayerShortPathInp(RecurrentMultiLayer):
    """
    Similar to the RecurrentMultiLayerShortPath class, just that the input
    is fed into the last layer of the MLP (similar to
    RecurrentMultiLayerInp).
    """

    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        self.W_shortp = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                        self.n_hids[dx],
                                        self.sparsity[dx],
                                        self.scale[dx],
                                        rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))

            if dx > 0:
                W_shp = self.init_fn[dx](self.n_hids[self.n_layers-1],
                                         self.n_hids[dx],
                                         self.sparsity[dx],
                                         self.scale[dx],
                                         rng=self.rng)
                self.W_shortp.append(theano.shared(value=W_shp,
                                               name='W_s%d_%s'%(dx,self.name)))
            if dx < self.n_layers-1:
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx](self.n_hids[dx],
                                       self.bias_scale[dx],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] + [x for x in self.b_hhs] +\
                [x for x in self.W_shortp]
        self.restricted_params = [x for x in self.params]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.nW_shortp = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_shortp]

            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs] + [x for x in self.nW_shortp]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]



    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs, self.nW_hhs)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs, self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
            W_shp = [(x+y) for x, y in zip(self.W_shortp, self.nW_shortp)]
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
            W_shp = self.W_shortp
        h = self.activation[0](TT.dot(state_before,
                                      W_hhs[0])+b_hhs[0])
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout
        rval += [h]
        for dx in xrange(1, self.n_layers-1):
            h = self.activation[dx](TT.dot(h,
                                           W_hhs[dx])+
                                    TT.dot(state_before,
                                           W_shp[dx-1])+b_hhs[dx])
            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        h = self.activation[-1](TT.dot(h, W_hhs[-1]) +
                                TT.dot(state_before, W_shp[-1])+state_below)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval +=[h]
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval += [h]
        return rval

class RecurrentMultiLayerShortPathInpAll(RecurrentMultiLayer):
    """
    Similar to RecurrentMultiLayerShortPathInp class, just that the input is
    fed to all layers of the MLP depicting the deep transition between h_tm1
    to h_t.
    """
    def _init_params(self):
        self.W_hhs = []
        self.W_shortp = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                        self.n_hids[dx],
                                        self.sparsity[dx],
                                        self.scale[dx],
                                        rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))

            if dx > 0:
                W_shp = self.init_fn[dx](self.n_hids[self.n_layers-1],
                                         self.n_hids[dx],
                                         self.sparsity[dx],
                                         self.scale[dx],
                                         rng=self.rng)
                self.W_shortp.append(theano.shared(value=W_shp,
                                               name='W_s%d_%s'%(dx,self.name)))
        self.params = [x for x in self.W_hhs] +\
                [x for x in self.W_shortp]

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]

        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nW_shortp = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_shortp]

            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nW_shortp]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]


    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs,self.nW_hhs)]
            W_shp = [(x+y) for x, y in zip(self.W_shortp,self.nW_shortp)]
        else:
            W_hhs = self.W_hhs
            W_shp = self.W_shortp
        def slice_state_below(dx, sb = state_below):
            st = 0
            for p in xrange(dx):
                st += self.n_hids[p]
            ed = st + self.n_hids[dx]
            if sb.ndim == 1:
                return sb[st:ed]
            else:
                return sb[:,st:ed]


        h = self.activation[0](TT.dot(state_before, W_hhs[0]) + slice_state_below(0))

        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)

        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval += [h]
        for dx in xrange(1, self.n_layers):
            h = self.activation[dx](TT.dot(h, W_hhs[dx]) +
                                    TT.dot(state_before, W_shp[dx-1]) +
                                    slice_state_below(dx))

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

class RecurrentLayer(Layer):
    """
        Standard recurrent layer with gates.
        See arXiv verion of our paper.
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 gating = False,
                 reseting = False,
                 gater_activation = TT.nnet.sigmoid,
                 reseter_activation = TT.nnet.sigmoid,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type gating: bool
        :param gating: If true, an update gate is used

        :type reseting: bool
        :param reseting: If true, a reset gate is used

        :type gater_activation: string or function
        :param name: The activation function of the update gate

        :type reseter_activation: string or function
        :param name: The activation function of the reset gate

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(gater_activation) is str or type(gater_activation) is unicode:
            gater_activation = eval(gater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        self.gating = gating
        self.reseting = reseting
        self.gater_activation = gater_activation
        self.reseter_activation = reseter_activation

        assert rng is not None, "random number generator should not be empty!"

        super(RecurrentLayer, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        if self.gating:
            self.G_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="G_%s"%self.name)
            self.params.append(self.G_hh)
        if self.reseting:
            self.R_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="R_%s"%self.name)
            self.params.append(self.R_hh)

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hh = theano.shared(self.W_hh.get_value()*0, name='noise_'+self.W_hh.name)
            self.noise_params = [self.nW_hh]
            if self.gating:
                self.nG_hh = theano.shared(self.G_hh.get_value()*0, name='noise_'+self.G_hh.name)
                self.noise_params += [self.nG_hh]
            if self.reseting:
                self.nR_hh = theano.shared(self.R_hh.get_value()*0, name='noise_'+self.R_hh.name)
                self.noise_params += [self.nR_hh]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def step_fprop(self,
                   state_below,
                   mask = None,
                   state_before = None,
                   gater_below = None,
                   reseter_below = None,
                   use_noise=True,
                   no_noise_bias = False,
                   W_hh_=None, G_hh_=None,
                   R_hh_=None):
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

        :type gater_below: theano variable
        :param gater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        rval = []
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

        if self.weight_noise and use_noise and self.noise_params:
            W_hh += self.nW_hh
            if self.gating:
                G_hh += self.nG_hh
            if self.reseting:
                R_hh += self.nR_hh
        else:
            W_hh = self.W_hh
            if self.gating:
                G_hh = self.G_hh
            if self.reseting:
                R_hh = self.R_hh

        # Reset gate:
        # optionally reset the hidden state.
        if self.reseting and reseter_below:
            reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                    reseter_below)
            reseted_state_before = reseter * state_before
        else:
            reseted_state_before = state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        if self.gating and gater_below:
            gater = self.gater_activation(TT.dot(state_before, G_hh) +
                    gater_below)
            h = gater * h + (1-gater) * state_before

        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before

        return h

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False
             ):

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
            if gater_below:
                gater_below = gater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        non_seqs = [self.W_hh, self.G_hh, self.R_hh]

        # FIXME: Find a way to clean this up
        if self.reseting and reseter_below:
            if self.gating and gater_below:
                if mask:
                    inps = [state_below, mask, gater_below, reseter_below]
                    fn = lambda x,y,g,r,z, whh, ghh, rhh: self.step_fprop(x,y,z, gater_below=g,
                                                       reseter_below=r, use_noise=use_noise,
                                                       W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, gater_below, reseter_below]
                    fn = lambda tx, tg,tr, ty, whh, ghh, rhh: self.step_fprop(tx, None, ty, gater_below=tg,
                                                        reseter_below=tr,
                                                        use_noise=use_noise,
                                                        W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                        no_noise_bias=no_noise_bias)
            else:
                if mask:
                    inps = [state_below, mask, reseter_below]
                    fn = lambda x,y,r,z, whh, ghh, rhh : self.step_fprop(x,y,z, use_noise=use_noise,
                                                        reseter_below=r,
                                                        W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                        no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, reseter_below]
                    fn = lambda tx,tr,ty, whh, ghh, rhh: self.step_fprop(tx, None, ty,
                                                        reseter_below=tr,
                                                        use_noise=use_noise,
                                                        W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                        no_noise_bias=no_noise_bias)
        else:
            if self.gating and gater_below:
                if mask:
                    inps = [state_below, mask, gater_below]
                    fn = lambda x,y,g,z, whh, ghh, rhh : self.step_fprop(x,y,z, gater_below=g, use_noise=use_noise,
                                                                         W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                                         no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, gater_below]
                    fn = lambda tx, tg, ty, whh, ghh, rhh: self.step_fprop(tx, None, ty, gater_below=tg,
                                                                           W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                                           use_noise=use_noise,
                                                                           no_noise_bias=no_noise_bias)
            else:
                if mask:
                    inps = [state_below, mask]
                    fn = lambda x, y, z, whh, ghh, rhh : self.step_fprop(x,y,z, use_noise=use_noise,
                                                                         W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                                         no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below]
                    fn = lambda tx, ty, whh, ghh, rhh: self.step_fprop(tx, None, ty,
                                                                       W_hh_=whh, G_hh_=ghh, R_hh_=rhh,
                                                                       use_noise=use_noise,
                                                                       no_noise_bias=no_noise_bias)
        rval, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = [init_state],
                        non_sequences=non_seqs,
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        strict=True,
                        n_steps = nsteps)
        new_h = rval
        self.out = rval
        self.rval = rval
        self.updates =updates

        return self.out


class LSTMLayer(Layer):
    """
        Standard LSTM Layer
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 name=None,
                 **kwargs):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit

        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.
        """
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout

        assert rng is not None, "random number generator should not be empty!"

        super(LSTMLayer, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hi = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Whi_%s"%self.name)
        self.params = [self.W_hi]
        self.W_ci = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wci_%s"%self.name)
        self.params += [self.W_ci]
        self.W_hf = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Whf_%s"%self.name)
        self.params += [self.W_hf]
        self.W_cf = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wcf_%s"%self.name)
        self.params += [self.W_cf]
        self.W_hc = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wcf_%s"%self.name)
        self.params += [self.W_hc]
        self.W_ho = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wcf_%s"%self.name)
        self.params += [self.W_ho]
        self.W_co = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wcf_%s"%self.name)
        self.params += [self.W_co]

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hi = theano.shared(self.W_hi.get_value()*0, name='noise_'+self.W_hi.name)
            self.noise_params = [self.nW_hi]
            self.nW_ci = theano.shared(self.W_ci.get_value()*0, name='noise_'+self.W_ci.name)
            self.noise_params += [self.nW_ci]
            self.nW_hf = theano.shared(self.W_hf.get_value()*0, name='noise_'+self.W_hf.name)
            self.noise_params += [self.nW_hf]
            self.nW_cf = theano.shared(self.W_cf.get_value()*0, name='noise_'+self.W_cf.name)
            self.noise_params += [self.nW_cf]
            self.nW_hc = theano.shared(self.W_hc.get_value()*0, name='noise_'+self.W_hc.name)
            self.noise_params += [self.nW_hc]
            self.nW_ho = theano.shared(self.W_ho.get_value()*0, name='noise_'+self.W_ho.name)
            self.noise_params += [self.nW_ho]
            self.nW_co = theano.shared(self.W_co.get_value()*0, name='noise_'+self.W_co.name)
            self.noise_params += [self.nW_co]

            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def _get_slice_below(self, state_below, to='cell'):
        if to == 'cell':
            offset = 0
        elif to == 'input':
            offset = 1 * self.n_hids
        elif to == 'output':
            offset = 2 * self.n_hids
        elif to == 'forget':
            offset = 3 * self.n_hids
        else:
            raise Warning('Unknown gate/cell types')

        if state_below.ndim == 3:
            return state_below[:,:,offset:offset+self.n_hids]
        if state_below.ndim == 2:
            return state_below[:,offset:offset+self.n_hids]
        return state_below[offset:offset+self.n_hids]

    def _get_slice_before(self, state_before, fr='cell'):
        if fr == 'cell':
            offset = self.n_hids
        elif fr == 'hidden':
            offset = 0
        else:
            raise Warning('Unknown cell/gate types')

        if state_before.ndim == 2:
            return state_before[:,offset:offset+self.n_hids]
        return state_before[offset:offset+self.n_hids]

    def step_fprop(self,
                   state_below,
                   mask = None,
                   state_before = None,
                   use_noise=True,
                   no_noise_bias = False,
                   **kwargs):
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

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hi = self.W_hi + self.nW_hi
            W_ci = self.W_ci + self.nW_ci
            W_hf = self.W_hf + self.nW_hf
            W_cf = self.W_cf + self.nW_cf
            W_hc = self.W_hc + self.nW_hc
            W_ho = self.W_ho + self.nW_ho
            W_co = self.W_co + self.nW_co
        else:
            W_hi = self.W_hi
            W_ci = self.W_ci
            W_hf = self.W_hf
            W_cf = self.W_cf
            W_hc = self.W_hc
            W_ho = self.W_ho
            W_co = self.W_co

        # input gate
        ig = TT.nnet.sigmoid(self._get_slice_below(state_below,'input') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_hi)  +
                TT.dot(self._get_slice_before(state_before,'cell'), W_ci))

        # forget gate
        fg = TT.nnet.sigmoid(self._get_slice_below(state_below,'forget') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_hf)  +
                TT.dot(self._get_slice_before(state_before,'cell'), W_cf))

        # cell
        cc = fg * self._get_slice_before(state_before,'cell') +  \
            ig * self.activation(self._get_slice_below(state_below,'cell') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_hc))

        # output gate
        og = TT.nnet.sigmoid(self._get_slice_below(state_below,'output') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_ho)  +
                TT.dot(cc, W_co))

        # hidden state
        hh = og * self.activation(cc)

        if hh.ndim == 2:
            h = TT.concatenate([hh, cc], axis=1)
        else:
            h = TT.concatenate([hh, cc], axis=0)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        return h

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False,
              **kwargs
             ):

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
            state_below = state_below.reshape((nsteps, batch_size, state_below.shape[-1]))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids * 2)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids * 2)

        if mask:
            inps = [state_below, mask]
            fn = lambda x,y,z : self.step_fprop(x,y,z, use_noise=use_noise,
                                               no_noise_bias=no_noise_bias)
        else:
            inps = [state_below]
            fn = lambda tx, ty: self.step_fprop(tx, None, ty,
                                                use_noise=use_noise,
                                                no_noise_bias=no_noise_bias)

        rval, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = [init_state],
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        n_steps = nsteps)
        new_h = rval
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out




class LSTMLayerIlya(Layer):
    """
        LSTM Layer without cell to gates connections.
        In the way that Ilya Sutskever uses.
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 name=None,
                 **kwargs):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit

        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.
        """
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout

        assert rng is not None, "random number generator should not be empty!"
        super(LSTMLayerIlya, self).__init__(self.n_hids,
                                            self.n_hids,
                                            rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hi = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Whi_%s"%self.name)
        self.params = [self.W_hi]

        self.b_i = theano.shared(
                self.bias_fn(self.n_hids,
                             self.bias_scale,
                             self.rng),
                name="b_i_%s"%(self.name))
        self.params += [self.b_i]

        self.W_hc = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Whc_%s"%self.name)
        self.params += [self.W_ci]

        self.b_c = theano.shared(
                self.bias_fn(self.n_hids,
                             self.bias_scale,
                             self.rng),
                name="b_c_%s"%(self.name))
        self.params += [self.b_o]

        self.W_hf = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Whf_%s"%self.name)
        self.params += [self.W_hf]

        self.b_f = theano.shared(
                self.bias_fn(self.n_hids,
                             self.bias_scale,
                             self.rng),
                name="b_f_%s"%(self.name))
        self.params += [self.b_o]

        self.W_ho = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Who_%s"%self.name)
        self.params += [self.W_ho]

        self.b_o = theano.shared(
                self.bias_fn(self.n_hids,
                             self.bias_scale,
                             self.rng),
                name="b_o_%s"%(self.name))
        self.params += [self.b_o]

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]

        if self.weight_noise:
            self.nW_hi = theano.shared(self.W_hi.get_value()*0, name='noise_'+self.W_hi.name)
            self.noise_params = [self.nW_hi]
            self.nW_hf = theano.shared(self.W_hf.get_value()*0, name='noise_'+self.W_hf.name)
            self.noise_params += [self.nW_hf]
            self.nW_hc = theano.shared(self.W_hc.get_value()*0, name='noise_'+self.W_hc.name)
            self.noise_params += [self.nW_hc]
            self.nW_ho = theano.shared(self.W_ho.get_value()*0, name='noise_'+self.W_ho.name)
            self.noise_params += [self.nW_ho]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]

    def _get_slice_below(self, state_below, to='cell'):
        if to == 'cell':
            offset = 0
        elif to == 'input':
            offset = 1 * self.n_hids
        elif to == 'output':
            offset = 2 * self.n_hids
        elif to == 'forget':
            offset = 3 * self.n_hids
        else:
            raise Warning('Unknown gate/cell types')

        if state_below.ndim == 3:
            return state_below[:,:,offset:offset+self.n_hids]
        if state_below.ndim == 2:
            return state_below[:,offset:offset+self.n_hids]
        return state_below[offset:offset+self.n_hids]

    def _get_slice_before(self, state_before, fr='cell'):
        if fr == 'cell':
            offset = self.n_hids
        elif fr == 'hidden':
            offset = 0
        else:
            raise Warning('Unknown cell/gate types')

        if state_before.ndim == 2:
            return state_before[:,offset:offset+self.n_hids]

        return state_before[offset:offset+self.n_hids]

    def step_fprop(self,
                   state_below,
                   mask = None,
                   state_before = None,
                   use_noise=True,
                   no_noise_bias = False,
                   **kwargs):
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

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hi = self.W_hi + self.nW_hi
            W_hf = self.W_hf + self.nW_hf
            W_hc = self.W_hc + self.nW_hc
            W_ho = self.W_ho + self.nW_ho
        else:
            W_hi = self.W_hi
            W_hf = self.W_hf
            W_hc = self.W_hc
            W_ho = self.W_ho

        b_i = self.b_i
        b_f = self.b_f
        b_c = self.b_c
        b_o = self.b_o

        # input gate
        ig = TT.nnet.sigmoid(self._get_slice_below(state_below,'input') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_hi) + b_i)

        # forget gate
        fg = TT.nnet.sigmoid(self._get_slice_below(state_below,'forget') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_hf) + b_f)

        # cell
        cc = fg * self._get_slice_before(state_before,'cell') +  \
            ig * self.activation(self._get_slice_below(state_below,'cell') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_hc) + b_c)

        # output gate
        og = TT.nnet.sigmoid(self._get_slice_below(state_below,'output') +
                TT.dot(self._get_slice_before(state_before,'hidden'), W_ho) + b_o)

        # hidden state
        hh = og * self.activation(cc)

        if hh.ndim == 2:
            h = TT.concatenate([hh, cc], axis=1)
        else:
            h = TT.concatenate([hh, cc], axis=0)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        return h

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False,
              **kwargs):

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
            state_below = state_below.reshape((nsteps, batch_size, state_below.shape[-1]))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids * 2)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids * 2)

        if mask:
            inps = [state_below, mask]
            fn = lambda x,y,z : self.step_fprop(x, y, z,
                                                use_noise=use_noise,
                                                no_noise_bias=no_noise_bias)
        else:
            inps = [state_below]
            fn = lambda tx, ty: self.step_fprop(tx, None, ty,
                                                use_noise=use_noise,
                                                no_noise_bias=no_noise_bias)

        rval, updates = theano.scan(fn,
                                    sequences = inps,
                                    outputs_info = [init_state],
                                    name='layer_%s'%self.name,
                                    profile=self.profile,
                                    truncate_gradient = truncate_gradient,
                                    n_steps = nsteps)

        new_h = rval
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out


#class HierarchicalBidirectionalRecurrentLayer(Layer):
class DoubleRecurrentLayer(Layer):
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 gater_activation = TT.nnet.sigmoid,
                 reseter_activation = TT.nnet.sigmoid,
                 gating=True,
                 reseting=True,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type gater_activation: string or function
        :param name: The activation function of the update gate

        :type reseter_activation: string or function
        :param name: The activation function of the reset gate

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(gater_activation) is str or type(gater_activation) is unicode:
            gater_activation = eval(gater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        self.gater_activation = gater_activation
        self.reseter_activation = reseter_activation

        assert rng is not None, "random number generator should not be empty!"

        #super(HierarchicalBidirectionalRecurrentLayer, self).__init__(self.n_hids,
        super(DoubleRecurrentLayer, self).__init__(self.n_hids,
                                                   self.n_hids, rng,
                                                   name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        self.G_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="G_%s"%self.name)
        self.params.append(self.G_hh)
        self.R_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="R_%s"%self.name)
        self.params.append(self.R_hh)

        self.Wrev_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wrev_%s"%self.name)
        self.params += [self.Wrev_hh]
        self.Grev_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Grev_%s"%self.name)
        self.params.append(self.Grev_hh)
        self.Rrev_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Rrev_%s"%self.name)
        self.params.append(self.Rrev_hh)

        self.Wgg_fwd = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wgg_fwd_%s"%self.name)
        self.params += [self.Wgg_fwd]

        self.Wgg_rev = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="Wgg_rev_%s"%self.name)
        self.params += [self.Wgg_rev]

        self.W2_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W2_%s"%self.name)
        self.params.append(self.W2_hh)
        self.U2_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="U2_%s"%self.name)
        self.params.append(self.U2_hh)
        self.V2_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="V2_%s"%self.name)
        self.params.append(self.V2_hh)
        self.G2_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="G2_%s"%self.name)
        self.params.append(self.G2_hh)
        self.R2_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="R2_%s"%self.name)
        self.params.append(self.R2_hh)

        self.S_h = theano.shared(
                                 eval('numpy.'+theano.config.floatX)(0.01 * numpy.random.rand(self.n_hids)),
                                 #sample_weights_classic(self.n_hids, 1, -1, 0.01, rng=self.rng),
                                 name='S_%s'%self.name)
        self.params.append(self.S_h)

        self.W_att = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_att_%s"%self.name)
        self.params.append(self.W_att)

        self.U_att = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="U_att_%s"%self.name)
        self.params.append(self.U_att)

        self.V_att = theano.shared(
                                   eval('numpy.'+theano.config.floatX)(0.01 * numpy.random.rand(self.n_hids)),
                                   name='V_att_%s'%self.name)
        self.params.append(self.V_att)

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]

        if self.weight_noise:
            self.nW_hh = theano.shared(self.W_hh.get_value()*0, name='noise_'+self.W_hh.name)
            self.nG_hh = theano.shared(self.G_hh.get_value()*0, name='noise_'+self.G_hh.name)
            self.nR_hh = theano.shared(self.G_hh.get_value()*0, name='noise_'+self.G_hh.name)
            self.noise_params = [self.nW_hh,self.nG_hh,self.nR_hh]
            self.nW2_hh = theano.shared(self.W2_hh.get_value()*0, name='noise_'+self.W2_hh.name)
            self.nG2_hh = theano.shared(self.G2_hh.get_value()*0, name='noise_'+self.G2_hh.name)
            self.nR2_hh = theano.shared(self.G2_hh.get_value()*0, name='noise_'+self.G2_hh.name)
            self.noise_params += [self.nW2_hh,self.nG2_hh,self.nR2_hh]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False):

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
            if gater_below:
                gater_below = gater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        W_hh = self.W_hh
        G_hh = self.G_hh
        R_hh = self.R_hh
        W2_hh = self.W2_hh
        G2_hh = self.G2_hh
        R2_hh = self.R2_hh

        Wrev_hh = self.Wrev_hh
        Grev_hh = self.Grev_hh
        Rrev_hh = self.Rrev_hh

        Wgg_fwd = self.Wgg_fwd
        Wgg_rev = self.Wgg_rev

        U2_hh = self.U2_hh
        V2_hh = self.V2_hh
        W_att = self.W_att
        U_att = self.U_att
        V_att = self.V_att
        S_h = self.S_h

        def _scan1(state_below, mask, state_before, gater_below, reseter_below,
                   use_noise=True, no_noise_bias=False, R_hh_=None, W_hh_=None, G_hh_=None):

            reseter = self.reseter_activation(TT.dot(state_before, R_hh_) + reseter_below)
            reseted_state_before = reseter * state_before

            # Feed the input to obtain potential new state.
            preactiv = TT.dot(reseted_state_before, W_hh_) + state_below
            h = self.activation(preactiv)

            gater = self.gater_activation(TT.dot(state_before, G_hh_) + gater_below)
            h = gater * h + (1-gater) * state_before

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)

            if mask is not None:
                if h.ndim ==2 and mask.ndim==1:
                    mask = mask.dimshuffle(0,'x')
                h = mask * h + (1-mask) * state_before

            return h

        non_seqs = [R_hh, W_hh, G_hh]

        def _scan1rev(state_below, mask, state_before,
                      gater_below, reseter_below,
                      use_noise=True, no_noise_bias=False,
                      Rrev_hh_=None, Wrev_hh_=None,
                      Grev_hh_=None):

            reseter = self.reseter_activation(TT.dot(state_before, Rrev_hh_) + reseter_below)
            reseted_state_before = reseter * state_before

            # Feed the input to obtain potential new state.
            preactiv = TT.dot(reseted_state_before, Wrev_hh_) + state_below
            h = self.activation(preactiv)

            gater = self.gater_activation(TT.dot(state_before, Grev_hh_) + gater_below)
            h = gater * h + (1-gater) * state_before

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)

            if mask is not None:
                if h.ndim ==2 and mask.ndim==1:
                    mask = mask.dimshuffle(0,'x')
                h = mask * h + (1-mask) * state_before

            return h

        non_seqs_rev = [Rrev_hh, Wrev_hh, Grev_hh]

        if mask:
            inps = [state_below, mask, gater_below, reseter_below]
            fn = lambda x, y, g, r, z, rhh, whh, ghh: _scan1(x, y, z, gater_below=g,
                                                         reseter_below=r,
                                                         use_noise=use_noise,
                                                         no_noise_bias=no_noise_bias,
                                                         R_hh_=rhh, W_hh_=whh,
                                                         G_hh_=ghh)
        else:
            inps = [state_below, gater_below, reseter_below]
            fn = lambda tx, tg, tr, ty, rhh, whh, ghh: _scan1(tx, None, ty, gater_below=tg,
                                                              reseter_below=tr,
                                                              use_noise=use_noise,
                                                              no_noise_bias=no_noise_bias,
                                                              R_hh_=rhh, W_hh_=whh,
                                                              G_hh_=ghh)

        rval, updates = theano.scan(fn,
                                    sequences = inps,
                                    non_sequences=non_seqs,
                                    outputs_info = [init_state],
                                    name='layer_%s'%self.name,
                                    profile=self.profile,
                                    truncate_gradient = truncate_gradient,
                                    n_steps = nsteps,
                                    strict=True)

        rval_fwd = rval

        if mask:
            mask_rev = mask[::-1]
        else:
            mask_rev = None

        state_below_rev = state_below[::-1]
        gater_below_rev = gater_below[::-1]
        reseter_below_rev = reseter_below[::-1]

        if mask_rev:
            inps = [state_below_rev, mask_rev, gater_below_rev, reseter_below_rev]
            fn = lambda x,y,g,r,z, rhh, whh, ghh: _scan1rev(x,y,z, gater_below=g, reseter_below=r,
                                                            use_noise=use_noise,
                                                            no_noise_bias=no_noise_bias,
                                                            Rrev_hh_=rhh,
                                                            Wrev_hh_=whh,
                                                            Grev_hh_=ghh)
        else:
            inps = [state_below_rev, gater_below_rev, reseter_below_rev]
            fn = lambda tx, tg,tr, ty, rhh, whh, ghh: _scan1rev(tx, None, ty, gater_below=tg,
                                                                reseter_below=tr,
                                                                use_noise=use_noise,
                                                                no_noise_bias=no_noise_bias,
                                                                Rrev_hh_=rhh,
                                                                Wrev_hh_=whh,
                                                                Grev_hh_=ghh)

        rval, updates_rev = theano.scan(fn,
                                        sequences = inps,
                                        non_sequences = non_seqs_rev,
                                        outputs_info = [init_state],
                                        name = 'layer_%s' % self.name,
                                        profile = self.profile,
                                        truncate_gradient = truncate_gradient,
                                        n_steps = nsteps,
                                        strict = True)

        updates += updates_rev
        rval_rev = rval[::-1]

        state_below = TT.dot(rval_fwd, Wgg_fwd) + TT.dot(rval_rev, Wgg_rev)
        state_below_att = TT.dot(state_below, W_att)

        def _scan2(mask, state_before, beta_before,
                   use_noise=True, no_noise_bias = False, U_att_=None,
                   V_att_=None, R2_hh_=None, U2_hh_=None, W2_hh_=None,
                   G2_hh_=None, V2_hh_=None, S_h_=None, W_att=None,
                   Wgg_rev=None, Wgg_fwd=None, Wrev_hh=None, Rrev_hh=None,
                   Grev_hh=None, rval_fwd=None, rval_rev=None):

            # attention
            state_before_att = TT.dot(state_before, U_att_)
            if state_below_att.ndim == 3:
                att = TT.tanh(state_below_att + state_before_att[None,:,:])
            else:
                att = TT.tanh(state_below_att + state_before_att[None,:])

            att = TT.exp(TT.dot(att, V_att_))
            att = att / att.sum(0, keepdims=True)

            if state_below.ndim == 3:
                real_below = (state_below * att[:,:,None]).sum(axis=0)
            else:
                real_below = (state_below * att[:,None]).sum(axis=0)

            # reset gate
            reseter = self.reseter_activation(TT.dot(state_before, R2_hh_) +
                                              TT.dot(real_below, U2_hh_))

            reseted_state_before = reseter * state_before

            # Feed the input to obtain potential new state.
            preactiv = TT.dot(reseted_state_before, U2_hh_) + TT.dot(real_below, W2_hh_)
            h = self.activation(preactiv)

            # update gate
            gater = self.gater_activation(TT.dot(state_before, G2_hh_) +
                                          TT.dot(real_below, V2_hh_))
            h = gater * h + (floatX(1) - gater) * state_before

            if mask is not None:
                if h.ndim ==2 and mask.ndim==1:
                    mask = mask.dimshuffle(0,'x')
                h = mask * h + (floatX(1) - mask) * state_before

            # stopping probability
            s = TT.nnet.sigmoid(TT.dot(h, S_h_))
            beta = beta_before * (floatX(1) - s)

            return h, beta

        non_seqs2 = [U_att, V_att, R2_hh, U2_hh, W2_hh, G2_hh, V2_hh, S_h, W_att, Wgg_rev, Wgg_fwd, rval_fwd, rval_rev]
        if mask:
            inps = [mask]
            fn = lambda y, z, b, uat, vat, r2hh, u2hh, w2hh, g2hh, v2hh, sh, watt, wggr, wggf, rvf, rvr: _scan2(y, z, b,
                    U_att_=uat, V_att_=vat, R2_hh_=r2hh,
                    U2_hh_=u2hh, W2_hh_=w2hh, G2_hh_=g2hh,
                    V2_hh_=v2hh, S_h_=sh, W_att=watt, Wgg_rev=wggr,
                    Wgg_fwd=wggf,
                    rval_fwd=rvf, rval_rev=rvr,
                    use_noise=use_noise,
                    no_noise_bias=no_noise_bias)
        else:
            inps = []
            fn = lambda y,b, uat, vat, r2hh, u2hh, w2hh, g2hh, v2hh, sh, watt, wggr, wggf, rvf, rvr: _scan2(None, y, b,
                                                                                                  U_att_=uat, V_att_=vat, R2_hh_=r2hh,
                                                                                                  U2_hh_=u2hh, W2_hh_=w2hh, G2_hh_=g2hh,
                                                                                                  V2_hh_=v2hh, S_h_=sh, W_att=watt,
                                                                                                  Wgg_rev=wggr, Wgg_fwd=wggf,
                                                                                                  rval_fwd=rvf, rval_rev=rvr,
                                                                                                  use_noise=use_noise, no_noise_bias=no_noise_bias)

        if (not isinstance(batch_size, int)
                or batch_size != 1):
            init_beta = TT.alloc(floatX(1), batch_size)
        else:
            init_beta = TT.alloc(floatX(1), 1)

        rval, updates2 = theano.scan(fn,
                                     sequences = inps,
                                     non_sequences=non_seqs2,
                                     outputs_info = [init_state, init_beta],
                                     name='layer2_%s' % self.name,
                                     profile = self.profile,
                                     truncate_gradient = truncate_gradient,
                                     n_steps = nsteps,
                                     strict=True)

        new_h = rval[0]
        betas = rval[1]
        updates += updates2

        if new_h.ndim == 3:
            rval[0] = new_h * betas[:,:,None]
        else:
            rval[0] = new_h * betas[:,None]

        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out
