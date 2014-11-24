def prototype_lm_state():

    state = {}

    state['seed'] = 1234
    state['level'] = 'DEBUG'

    #----------- DATA -----------

    state['min_lr'] = float(5e-7) 

    # TODO H5 the dataset and shuffle it
    state['target'] = '/data/lisatmp3/xukelvin/translation/fr/binarized_lm_text.fr.shuf.h5'

    state['indx_word'] = '/data/lisatmp3/xukelvin/translation/fr/ivocab.fr.pkl'
    state['word_indx'] = 'data/lisatmp3/xukelvin/translation/fr/vocab.fr.pkl'

    state['encoding'] = 'ascii'

    state['oov'] = 'UNK'
    # unknown 
    state['unk_sym'] = 1
    # end of sequence
    state['null_sym'] = 30000
    # total vocabulary size
    state['n_sym'] = state['null_sym'] + 1

    state['sort_k_batches'] = 20

    state['bs'] = 100

    state['use_infinite_loop'] = True
    state['seqlen'] = 50
    state['shuffle'] = False
    state['trim_batches'] = True

    #---------- MODEL Structure ---------

    state['bias'] = 0.

    # TODO 
    # from jointly learning to align paper
    state['rank_n_approx'] = 620
    state['dim'] = 1000

    # weight noise settings 
    state['weight_init_fn'] = 'sample_weights_classic'
    state['weight_scale'] = 0.01
    state['weight_noise'] = False 
    state['weight_noise_rec'] = False

    # no noise on biases 
    state['no_noise_bias'] = True

    state['weight_noise_amount'] = 0.01

    state['rank_n_activ'] = 'lambda x: x'

    state['rec_layer'] = 'RecurrentLayer'
    state['rec_gating'] = True
    state['rec_reseting'] = True
    state['rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'

    # related to the softmax 
    state['out_scale'] = .1
    state['out_bias_scale'] = -.5
    state['out_sparse'] = -1

    state['activ'] = 'lambda x: TT.tanh(x)'

    state['rec_weight_init_fn'] = 'sample_weights_orth'
    state['rec_weight_scale'] = 1.    

    state['check_first_word'] = True
    state['eps'] = 1e-10

    # ------ TRAINING METHOD ----

    # Bleu validation
    state['bleu_val_frequency'] = None

    # Turns on noise contrastive estimation instead maximum likelihood
    state['use_nce'] = False

    # Choose optimization algorithm
    state['algo'] = 'SGD_adadelta'

    # Early Stopping Stuff
    state['patience'] = 1
    state['lr'] = 1.
    state['minlr'] = 0

    # this should be always true
    state['carry_h0'] = True

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # A magic gradient clipping option that you should never change...
    state['cutoff_rescale_length'] = 0.

    # Loading / Saving
    # Prefix for the model, state and timing files
    state['prefix'] = '/data/lisatmp3/xukelvin/translation/fr_lm/lm_'
    # Specifies whether old model should be reloaded first
    state['reload'] = False
    # When set to 0 each new model dump will be saved in a new file
    state['overwrite'] = 1

    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*7
    # Error level to stop at
    state['minerr'] = -1

    # Reset data iteration every this many epochs
    state['reset'] = -1
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 10
    # Frequency of running hooks
    state['sampling_seed'] = 5
    state['hookFreq'] = 1000 
    # Validation frequency
    state['validFreq'] = 500
    # Model saving frequency (in minutes)
    state['saveFreq'] = 10

    # Sampling hook settings
    state['sample_steps'] = 10
    state['n_examples'] = 3

    # Raise exception if nan
    state['on_nan'] = 'raise'

    return state


def prototype_lm_state_en():
    state = prototype_lm_state()
    state['target'] = '/data/lisatmp3/xukelvin/translation/en_lm/binarized_wiki.en.shuf.h5'

    state['indx_word'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/tr-en_lm/ijoint_vocab.pkl'
    state['word_indx'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/tr-en_lm/joint_vocab.pkl'

    # index of 'the' in english 
    state['sampling_seed'] = 4

    state['prefix'] = 'lm_'

    return state

def prototype_lm_state_tr():    
    state = prototype_lm_state()
    state['target'] = '/data/lisatmp3/firatorh/languageModelling/corpora/tr_lm/train/binarized_text.tr.h5'
    
    state['indx_word'] = '/data/lisatmp3/firatorh/languageModelling/corpora/tr_lm/ivocab.pkl'
    state['word_indx'] = '/data/lisatmp3/firatorh/languageModelling/corpora/tr_lm/vocab.pkl'
    
    state['sampling_seed'] = 4
    
    state['prefix'] = 'trLM_'
    
    return state
