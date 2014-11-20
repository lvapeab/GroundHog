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
        Concatenate
from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator
from groundhog.utils import sample_zeros, sample_weights_orth, init_bias, sample_weights_classic
import groundhog.utils as utils
from experiments.nmt.language_model import LM_builder

logger = logging.getLogger(__name__)

class LM_wrapper(LM_Model):

    
    def __init__(self, 
                 lm_type,
                 state_file,
                 model_file,
                 rng):
        logger.debug("Employ external Language Model")
        
        # this must initialize language model and create its layers
        self.lm = eval(lm_type)(self.state_from_file(state_file), rng)
        self.n_hids = self.lm.get_n_hids()
        
        # this must build computational graph of language model
        self.lm.build()
        
        super(LM_wrapper,self).__init__(cost_layer = self.lm.train_model,
                                        sample_fn = self.lm.get_sampler(),
                                        valid_fn = None,
                                        noise_fn = None,
                                        weight_noise_amount = self.lm.state['weight_noise_amount'],
                                        indx_word=self.lm.state['indx_word'],
                                        indx_word_src=None,
                                        exclude_params_for_norm=None,
                                        rng = rng)
        # load language model, 
        self.load(model_file)
    
    def state_from_file(self, filename):
        '''
        load state pickle file into state dictionary
        '''
        return cPickle.load(open(filename, "rb"))
    
    def single_step_sampler(self):
        return self.lm.get_ss_sampler()

    def get_sampler(self):
        return self.lm.get_sampler()
    
    def ht_sampler(self):
        return self.lm.get_ht_sampler()
    