from groundhog.models import LM_Model
from groundhog.layers.basic import Model
from groundhog.layers import MultiLayer, \
       SoftmaxLayer, \
       LastState,\
       UnaryOp, \
       DropOp, \
       Operator, \
       Shift, \
       GaussianNoise, \
       SigmoidLayer, \
       RecurrentLayer

import numpy
import theano
import theano.tensor as TT
import os
import cPickle
import logging

logger = logging.getLogger(__name__)

class DummyLM(Model):

    def __init__(self, state, rng):
        '''
        initialize dummy language model
        '''
        
        logger.debug("DummyLM used as an external language model")
        
        self.state  = state
        self.rng    = rng
        self.params = []
        self.grad_scale = 1.
        
        # load state if it is a filename
        if type(state) is str and os.path.exists(state):
             self.load_state(state)
        
        state = self.state

        default_kwargs = dict(
            init_fn=self.state['weight_init_fn'],
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])
        embedder_kwargs = dict(default_kwargs)
        embedder_kwargs.update(dict(
            n_in=self.state['rank_n_approx'],
            n_hids=self.state['n_hids'],
            activation=['lambda x:x']))
        
        #### Word Embedding
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['n_sym_source'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='LM_approx_embdr',
            **default_kwargs)
        
        self.input_embedder = MultiLayer(
                self.rng,
                name='LM_input_embdr',
                **embedder_kwargs)
        
        self.update_embedder = MultiLayer(
                self.rng,
                learn_bias=False,
                name='LM_update_embdr',
                **embedder_kwargs)
        
        self.reset_embedder =  MultiLayer(
                self.rng,
                learn_bias=False,
                name='LM_reset_embdr',
                **embedder_kwargs)
        
        #### Gated Recurrent Layer
        self.rec = eval(state['rec_layer'])(
                    self.rng,
                    n_hids=state['n_hids'],
                    activation=self.state['activ'],
                    bias_scale=self.state['bias'],
                    init_fn=self.state['rec_weight_init_fn'],
                    scale=self.state['rec_weight_scale'],
                    weight_noise=self.state['weight_noise_rec'],
                    dropout=self.state['dropout_rec'],
                    gating=self.state['rec_gating'],
                    gater_activation=self.state['rec_gater'],
                    reseting=self.state['rec_reseting'],
                    reseter_activation=self.state['rec_reseter'],
                    name='LM_rec')
        
        #### Softmax Layer
        self.output_layer = SoftmaxLayer(
                self.rng,
                state['n_hids'],
                state['n_out'],
                scale=state['out_scale'],
                bias_scale=state['out_bias_scale'],
                init_fn="sample_weights_classic",
                weight_noise=state['weight_noise'],
                sparsity=state['out_sparse'],
                sum_over_time=True,
                name='LM_out')
        
        self.build()
        
        super(DummyLM, self).__init__(output_layer=self.train_model,
                                      sample_fn=self.sample_fn,
                                      indx_word=self.state['indx_word'],
                                      rng=rng)
        
        [self.params.append(p) for p in self.approx_embedder.params] 
        [self.params.append(p) for p in self.input_embedder.params]
        [self.params.append(p) for p in self.update_embedder.params]
        [self.params.append(p) for p in self.reset_embedder.params]
        [self.params.append(p) for p in self.rec.params]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        
        
    def build(self):
        '''
        build computational graph of LM 
        '''
        logger.debug("building DummyLM language model")
        
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')
        
        approx_embeddings = self.approx_embedder(self.x)
        
        input_signal = self.input_embedder(approx_embeddings)
        update_signal = self.update_embedder(approx_embeddings)
        reset_signal = self.reset_embedder(approx_embeddings)
        
        hidden_layer = self.rec(input_signal,
                                nsteps=self.x.shape[0],
                                batch_size=self.x.shape[1] if self.x.ndim == 2 else 1,
                                mask=self.x_mask,
                                gater_below=self.none_if_zero(update_signal),
                                reseter_below=self.none_if_zero(reset_signal),
                                use_noise=False)
        
        ht = LastState()(hidden_layer)
        if ht.out.ndim == 2:
            ht.out = ht.out[:,:self.state['n_hids']]
        else:
            ht.out = ht.out[:self.state['n_hids']]

        self.train_model = self.output_layer(ht.out).train(target=self.y,
                        scale=numpy.float32(1./self.state['seqlen']))
        
        
    def load_state(self, state_filename):
        '''
        load state pickle file into state dictionary
        '''
        self.state = cPickle.load(open(state_filename, "rb"))
        
        
    #### Sampling
    ##### single-step sampling
    def sample_fn(self,word_tm1, h_tm1):
        '''
        perform a single step of sampling
        '''
        
        x = TT.lvector('x')
        y = TT.lvector('y')
        h0 = theano.shared(numpy.zeros((self.state['n_hids'],), dtype='float32'))
        
        approx_embeddings = self.approx_embedder(x)
        
        input_signal = self.input_embedder(approx_embeddings)
        update_signals = self.update_embedders(approx_embeddings)
        reset_signals = self.reset_embedders(approx_embeddings)
        
        
        train_model = output_layer(rec_layer).train(target=y,
                        scale=numpy.float32(1./state['seqlen']))
        
        '''
        x_emb = self.emb_words(word_tm1, use_noise = False, one_step=True)
        h0 = self.rec(x_emb, state_before=h_tm1, one_step=True, use_noise=False)[-1]
        word = self.output_layer.get_sample(state_below=h0, temp=1.)
        return word, h0
        '''

    def none_if_zero(self, x):
        if x == 0:
            return None
        return x
    
    def save_params(self,filename):
        logger.debug("DummyLM used as an external language model")
    

if __name__ == "__main__":

    # create dummy state and model files here
    state = {}
    state['indx_word'] = "/data/lisatmp3/chokyun/mt/ivocab_target.pkl"
    state['null_sym_source'] = 30000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1
    state['weight_init_fn'] = 'sample_weights_classic'
    state['weight_scale'] = 0.01
    
    # Embedding Layer
    state['seed'] = 1234
    state['n_in'] = 30000 
    state['inp_nhids'] = '[500]'
    state['rank_n_approx'] = 0
    state['rank_n_activ'] = 'lambda x: x'
    state['inp_scale'] = .1
    state['inp_sparse'] = -1
    state['inp_bias'] = '[0.]'
        
    # Gated Recurrent Layer
    state['n_hids'] = 750
    state['rec_layer'] = 'RecurrentLayer'
    state['rec_gating'] = True
    state['rec_reseting'] = True
    state['rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['activ'] = 'lambda x: TT.tanh(x)'
    state['bias'] = 0.
    state['rec_weight_init_fn'] = 'sample_weights_orth'
    state['rec_weight_scale'] = 1.
    state['dropout_rec'] = 1.
    
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01
    
    state['seqlen'] = 50
    state['no_noise_bias'] = False
        
    # Softmax Layer
    state['n_out'] = 30000
    state['out_scale'] = .1
    state['out_bias_scale'] = -.5
    state['out_sparse'] = -1

    cPickle.dump(state, 
                 open( '/data/lisatmp3/firatorh/languageModelling/lm_state.pkl', "wb" ),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    
    rng = numpy.random.RandomState(state['seed'])
    dlm = DummyLM(state,rng)
    dlm.build()    
    dlm.save_params('/data/lisatmp3/firatorh/languageModelling/lm_model.npz')


