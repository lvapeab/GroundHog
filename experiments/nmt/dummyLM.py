from groundhog.models import LM_Model
from groundhog.layers.basic import Model

import numpy
import theano
import theano.tensor as TT

logger = logging.getLogger(__name__)

class DummyLM(Model):

    def __init__(self,
                 state = None):
        
        rng = numpy.random.RandomState(state['seed'])
        
        x = TT.lvector('x')
        y = TT.lvector('y')
        h0 = theano.shared(numpy.zeros((eval(state['nhids'])[-1],), dtype='float32'))

        #### Word Embedding
        emb_words = MultiLayer(
            rng,
            n_in=state['n_in'],
            n_hids=eval(state['inp_nhids']),
            activation=eval(state['inp_activ']),
            init_fn='sample_weights_classic',
            weight_noise=state['weight_noise'],
            rank_n_approx = state['rank_n_approx'],
            scale=state['inp_scale'],
            sparsity=state['inp_sparse'],
            learn_bias = True,
            bias_scale=eval(state['inp_bias']),
            name='emb_words')
        
        #### Deep Transition Recurrent Layer
        rec = eval(state['rec_layer'])(
                rng,
                eval(state['nhids']),
                activation = eval(state['rec_activ']),
                #activation = 'TT.nnet.sigmoid',
                bias_scale = eval(state['rec_bias']),
                scale=eval(state['rec_scale']),
                sparsity=eval(state['rec_sparse']),
                init_fn=eval(state['rec_init']),
                weight_noise=state['weight_noise'],
                name='rec')
        
        #### Stiching them together
        ##### (1) Get the embedding of a word
        x_emb = emb_words(x, no_noise_bias=state['no_noise_bias'])
        ##### (2) Embedding + Hidden State via DT Recurrent Layer
        reset = TT.scalar('reset')
        rec_layer = rec(x_emb, n_steps=x.shape[0],
                        init_state=h0*reset,
                        no_noise_bias=state['no_noise_bias'],
                        truncate_gradient=state['truncate_gradient'],
                        batch_size=1)
        
        ### Neural Implementation of the Operators: \lhd
        #### Softmax Layer
        output_layer = SoftmaxLayer(
            rng,
            eval(state['nhids'])[-1],
            state['n_out'],
            scale=state['out_scale'],
            bias_scale=state['out_bias_scale'],
            init_fn="sample_weights_classic",
            weight_noise=state['weight_noise'],
            sparsity=state['out_sparse'],
            sum_over_time=True,
            name='out')
        
        train_model = output_layer(rec_layer).train(target=y,
                        scale=numpy.float32(1./state['seqlen']))
        
        super(DummyLM, self).__init__(output_layer=cost_layer,
                                      sample_fn=sample_fn,
                                      indx_word=indx_word,
                                      indx_word_src=indx_word_src,
                                      rng=rng)
        
    #### Sampling
    ##### single-step sampling
    def sample_fn(self,word_tm1, h_tm1):
        x_emb = self.emb_words(word_tm1, use_noise = False, one_step=True)
        h0 = self.rec(x_emb, state_before=h_tm1, one_step=True, use_noise=False)[-1]
        word = self.output_layer.get_sample(state_below=h0, temp=1.)
        return word, h0
