def prototype_state():
    state = {}

    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    # ----- DATA -----
    # (all Nones in this section are placeholders for required values)

    # Source sequences (must be singleton list for backward compatibility)
    state['source'] = [None]
    # Target sequences (must be singleton list for backard compatiblity)
    state['target'] = [None]
    # index -> word dict for the source language
    state['indx_word'] = None
    # index -> word dict for the target language
    state['indx_word_target'] = None
    # word -> index dict for the source language
    state['word_indx'] = None
    # word -> index dict for the target language
    state['word_indx_trgt'] = None

    # ----- VOCABULARIES -----
    # (all Nones in this section are placeholders for required values)

    # A string representation for the unknown word placeholder for both language
    state['oov'] = 'UNK'
    # These are unknown word placeholders
    state['unk_sym_source'] = 1
    state['unk_sym_target'] = 1
    # These are end-of-sequence marks
    state['null_sym_source'] = None
    state['null_sym_target'] = None
    # These are vocabulary sizes for the source and target languages
    state['n_sym_source'] = None
    state['n_sym_target'] = None

    # ----- MODEL STRUCTURE -----

    # The components of the annotations produced by the Encoder
    state['last_forward'] = True
    state['last_backward'] = False
    state['forward'] = False
    state['backward'] = False
    # Turns on "search" mechanism
    state['search'] = False
    # Turns on using the shortcut from the previous word to the current one
    state['bigram'] = True
    # Turns on initialization of the first hidden state from the annotations
    state['bias_code'] = True
    # Turns on using the context to compute the next Decoder state
    state['decoding_inputs'] = True
    # Turns on an intermediate maxout layer in the output
    state['deep_out'] = True
    # Heights of hidden layers' stacks in encoder and decoder
    # WARNING: has not been used for quite while and most probably
    # doesn't work...
    state['encoder_stack'] = 1
    state['decoder_stack'] = 1
    # Use the top-most recurrent layer states as annotations
    # WARNING: makes sense only for hierachical RNN which
    # are in fact currently not supported
    state['take_top'] = True
    # Activates age old bug fix - should always be true
    state['check_first_word'] = True

    state['eps'] = 1e-10

    # Deep MLP parameters for deep attention mechanism
    state['deep_attention'] = None

    # ----- MODEL COMPONENTS -----

    # Low-rank approximation activation function
    state['rank_n_activ'] = 'lambda x: x'
    # Hidden-to-hidden activation function
    state['activ'] = 'lambda x: TT.tanh(x)'
    # Nonlinearity for the output
    state['unary_activ'] = 'Maxout(2)'

    # Hidden layer configuration for the forward encoder
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['enc_rec_gating'] = True
    state['enc_rec_reseting'] = True
    state['enc_rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['enc_rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    # Hidden layer configuration for the decoder
    state['dec_rec_layer'] = 'RecurrentLayer'
    state['dec_rec_gating'] = True
    state['dec_rec_reseting'] = True
    state['dec_rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['dec_rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    # Default hidden layer configuration, which is effectively used for
    # the backward RNN
    # TODO: separate back_enc_ configuration and convert the old states
    # to have it
    state['rec_layer'] = 'RecurrentLayer'
    state['rec_gating'] = True
    state['rec_reseting'] = True
    state['rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'

    # ----- SIZES ----

    # Dimensionality of hidden layers
    state['dim'] = 1000
    # Dimensionality of low-rank approximation
    state['rank_n_approx'] = 100
    # k for the maxout stage of output generation
    state['maxout_part'] = 2.

    # ----- WEIGHTS, INITIALIZATION -----

    # This one is bias applied in the recurrent layer. It is likely
    # to be zero as MultiLayer already has bias.
    state['bias'] = 0.

    # Weights initializer for the recurrent net matrices
    state['rec_weight_init_fn'] = 'sample_weights_orth'
    state['rec_weight_scale'] = 1.
    # Weights initializer for other matrices
    state['weight_init_fn'] = 'sample_weights_classic'
    state['weight_scale'] = 0.01

    # ----- BLEU VALIDATION OPTIONS ----
    # Location of the evaluation script
    state['bleu_script']=None#'/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    # Location of the validation set
    state['validation_set']=None#'/data/lisatmp3/firatorh/turkishParallelCorpora/compiled/tr-en/devSet/IWSLT14.TED.dev2010.tr-en.tr.xml.tok.seg'
    # boolean, whether or not to write the validation set to file
    state['output_validation_set'] = False
    # Location of the validation set output, if different
    # fom default
    state['validation_set_out'] = None
    # Location of what to compare the output translation to (gt)
    state['validation_set_grndtruth']=None#'/data/lisatmp3/firatorh/turkishParallelCorpora/compiled/tr-en/devSet/IWSLT14.TED.dev2010.tr-en.en.xml.tok'
    # Beam size during sampling
    state['beam_size'] = 20
    # Number of steps between every validation
    state['bleu_val_frequency'] = 20000
    # Character or word based BLEU calculation
    state['char_based_bleu'] = False
    # boolean, whether or not target words are segmented into suffixes
    state['target_words_segmented'] = False
    # source encoding
    state['source_encoding'] = 'ascii'
    # target encoding
    state['target_encoding'] = 'ascii'
    # start after this many iterations
    state['validation_burn_in'] = 10000

    # ---- REGULARIZATION -----

    # WARNING: dropout is not tested and probably does not work.
    # Dropout in output layer
    state['dropout'] = 0.5
    # Dropout in recurrent layers
    state['dropout_rec'] = 1.

    # WARNING: weight noise regularization is not tested
    # and most probably does not work.
    # Random weight noise regularization settings
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # A magic gradient clipping option that you should never change...
    state['cutoff_rescale_length'] = 0.

    # ----- TRAINING METHOD -----

    # Turns on noise contrastive estimation instead maximum likelihood
    state['use_nce'] = False

    # Choose optimization algorithm
    state['algo'] = 'SGD_adadelta'

    # Adadelta hyperparameters
    state['adarho'] = 0.95
    state['adaeps'] = 1e-6

    # Early stopping configuration
    # WARNING: was never changed during machine translation experiments,
    # as early stopping was not used.
    state['patience'] = 1
    state['lr'] = 1.
    state['minlr'] = 0

    # Batch size
    state['bs']  = 64
    # We take this many minibatches, merge them,
    # sort the sentences according to their length and create
    # this many new batches with less padding.
    state['sort_k_batches'] = 10

    # Maximum sequence length
    state['seqlen'] = 30
    # Turns on trimming the trailing paddings from batches
    # consisting of short sentences.
    state['trim_batches'] = True
    # Loop through the data
    state['use_infinite_loop'] = True
    # Start from a random entry
    state['shuffle'] = False

    # ----- TRAINING PROCESS -----

    # Prefix for the model, state and timing files
    state['prefix'] = 'phrase_'
    # Specifies whether old model should be reloaded first
    state['reload'] = True
    # When set to 0 each new model dump will be saved in a new file
    state['overwrite'] = 1

    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    # Reset data iteration every this many epochs
    state['reset'] = -1
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 1
    # Frequency of running hooks
    state['hookFreq'] = 13
    # Validation frequency
    state['validFreq'] = 500
    # Model saving frequency (in minutes)
    state['saveFreq'] = 10

    # Sampling hook settings
    state['n_samples'] = 3
    state['n_examples'] = 3

    # Raise exception if nan
    state['on_nan'] = 'raise'

    return state

def prototype_phrase_state():
    """This prototype is the configuration used in the paper
    'Learning Phrase Representations using RNN Encoder-Decoder
    for  Statistical Machine Translation' """

    state = prototype_state()

    state['source'] = ["/data/lisatmp3/bahdanau/shuffled/phrase-table.en.h5"]
    state['target'] = ["/data/lisatmp3/bahdanau/shuffled/phrase-table.fr.h5"]
    state['indx_word'] = "/data/lisatmp3/chokyun/mt/ivocab_source.pkl"
    state['indx_word_target'] = "/data/lisatmp3/chokyun/mt/ivocab_target.pkl"
    state['word_indx'] = "/data/lisatmp3/chokyun/mt/vocab.en.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/bahdanau/vocab.fr.pkl"

    state['null_sym_source'] = 15000
    state['null_sym_target'] = 15000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    return state

def prototype_encdec_state():
    """This prototype is the configuration used to train the RNNenc-30 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_state()

    baseDir='/data/lisatmp3/firatorh/nmt/tr-en_lm/'
    state['target'] = [baseDir + 'binarized_text.en.shuf.h5']
    state['source'] = [baseDir + 'binarized_text.tr.shuf.h5']
    state['indx_word'] = baseDir + 'ivocab.tr.pkl'
    state['indx_word_target'] = baseDir + 'ijoint_vocab.pkl'
    state['word_indx'] = baseDir + 'vocab.tr.pkl'
    state['word_indx_trgt'] = baseDir + 'joint_vocab.pkl'

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 30
    state['bs']  = 80

    state['dim'] = 1000
    state['rank_n_approx'] = 620

    state['prefix'] = 'encdec_'

    return state

def prototype_search_state():
    """This prototype is the configuration used to train the RNNsearch-50 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_encdec_state()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'

    state['deep_attention']= False
    state['deep_attention_n_hids']= [1500,1500]
    state['deep_attention_acts']= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) ']

    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = 'deepAttention_'

    return state

def prototype_phrase_lstm_state():
    state = prototype_phrase_state()

    state['enc_rec_layer'] = 'LSTMLayer'
    state['enc_rec_gating'] = False
    state['enc_rec_reseting'] = False
    state['dec_rec_layer'] = 'LSTMLayer'
    state['dec_rec_gating'] = False
    state['dec_rec_reseting'] = False
    state['dim_mult'] = 4
    state['prefix'] = 'phrase_lstm_'

    return state

def prototype_search_state_with_LM_TEST():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = 'searchWithLM_TEST_'

    return state

def prototype_search_state_with_LM_tr_en():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = 'searchWithLM0_'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/searchWithLM0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_test_prototype_eos20():
    """This prototype is the configuration used to train the RNNsearch-50 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['cutoff'] = 1.0
    state['hookFreq'] =200
    state['algo'] = 'SGD_rmsprop'

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = '/data/lisatmp3/xukelvin/tmp/joint_eos20/search_test_'
    return state


def prototype_search_state_with_LM_zh_en():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/binarized_text.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/ijoint_vocab.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/joint_vocab.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/searchWithLM0_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/searchWithLM0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_tr_en_without_LM():

    state = prototype_encdec_state()

    state['include_lm'] = False
    state['reload_lm'] = False
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 45

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/unionWithoutLM/searchWithoutLM_'

    baseDir='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/unionWithoutLM/'
    state['target'] = [baseDir + 'binarized_text.en.shuf.h5']
    state['source'] = [baseDir + 'binarized_text.tr.shuf.h5']
    state['indx_word'] = baseDir + 'ivocab.tr.pkl'
    state['indx_word_target'] = baseDir + 'iunion_dict.pkl'
    state['word_indx'] = baseDir + 'vocab.tr.pkl'
    state['word_indx_trgt'] = baseDir + 'union_dict.pkl'

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 30
    state['bs']  = 80

    state['dim'] = 1000
    state['rank_n_approx'] = 620

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/unionWithoutLM/union0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_tr_en_without_LM2():

    state = prototype_search_state_tr_en_without_LM()

    state['include_lm'] = False
    state['reload_lm'] = False
    state['mask_first_lm'] = False
    state['reload'] = False

    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/unionWithoutLM2/searchWithoutLM_'

    baseDir='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/unionWithoutLM2/'
    state['target'] = [baseDir + 'binarized_text.en.shuf.h5']
    state['source'] = [baseDir + 'binarized_text.tr.shuf.h5']
    state['indx_word'] = baseDir + 'ivocab.tr.pkl'
    state['indx_word_target'] = baseDir + 'iunion_dict.pkl'
    state['word_indx'] = baseDir + 'vocab.tr.pkl'
    state['word_indx_trgt'] = baseDir + 'union_dict.pkl'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/unionWithoutLM2/union0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_zh_en_without_LM():

    state = prototype_encdec_state()

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 45

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/binarized_text.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/iunion_dict.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/union_dict.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/searchWithoutLM_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionWithoutLM/searchWithoutLM_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_tr_en_with_shallow_LM():

    state = prototype_encdec_state()

    state['shallow_lm'] = True
    state['reload_lm'] = True
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = 'searchWithLM2_'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/searchWithLM2_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_zh_en_with_shallow_LM():

    state = prototype_encdec_state()

    state['shallow_lm'] = True
    state['reload_lm'] = True
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/binarized_text.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/ijoint_vocab.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/joint_vocab.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/searchWithLM2_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/searchWithLM2_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_with_LM_tr_en_finetune():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['train_only_readout'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/finetune/combined0_'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out'] ='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/finetune/searchWithLMfinetune_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_tr_en_SANITY_CHECK():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/sanityCheck/searchWithZeroLM_'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/sanityCheck/searchWithZeroLM_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_tr_en_MASK():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = True
    state['train_only_readout'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/masking/combined0_'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/masking/combined0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_tr_en_MASK_TEST():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = True
    state['train_only_readout'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/masking/combinedTEST_'

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/masking/combinedTEST_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_tr_en_UNION():

    state = prototype_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = False
    state['train_only_readout'] = False

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['saveFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/union/union0_'


    baseDir='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/union/'
    state['target'] = [baseDir + 'binarized_text.en.shuf.h5']
    state['source'] = [baseDir + 'binarized_text.tr.shuf.h5']
    state['indx_word'] = baseDir + 'ivocab.tr.pkl'
    state['indx_word_target'] = baseDir + 'iunion_dict.pkl'
    state['word_indx'] = baseDir + 'vocab.tr.pkl'
    state['word_indx_trgt'] = baseDir + 'union_dict.pkl'

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 30
    state['bs']  = 80

    state['dim'] = 1000
    state['rank_n_approx'] = 620

    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/tr-en_lm/dev/IWSLT14.TED.dev2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/union/union0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_zh_en_UNION():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = False
    state['train_only_readout'] = False

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/binarized_text.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/iunion_dict.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/union_dict.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/union0_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/union0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = False
    state['train_only_readout'] = True
    state['random_readout'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =10
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/binarized_text.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/iunion_dict.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/union_dict.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/unionRnd_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/unionRnd_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_zh_en_UNION_RAND():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['train_only_readout'] = False

    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionRnd/unionRnd_'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionRnd/unionRnd_valOut.txt'
    state['bleu_val_frequency'] = 2500
    return state

def prototype_search_state_with_LM_zh_en_UNION_RAND_CNT():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['train_only_readout'] = False
    state['random_readout'] = False

    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRndCnt/unionFinetuneRndCnt_'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRndCnt/unionFinetuneRndCnt_valOut.txt'
    state['bleu_val_frequency'] = 2500
    return state


def prototype_search_state_with_LM_zh_en_MASK_TEST_UNION():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = True
    state['train_only_readout'] = True
    state['random_readout'] = True

    state['hookFreq'] =25

    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionMasking/maskingTest_'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionMasking/maskingTest_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_zh_en_UNION_TANH_FINETUNE():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = False
    state['train_only_readout'] = True
    state['random_readout'] = True

    state['hookFreq'] =50

    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRndTanh/unionTanh_'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRndTanh/unionTanh_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE_STARBUCKS():

    state = prototype_encdec_state()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['mask_first_lm'] = False
    state['train_only_readout'] = True
    state['random_readout'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =10
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/union/binarized_text.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/iunion_dict.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRnd/union_dict.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/gulcehrc/nmt/zh-en/trainedModels/unionFinetuneRnd/unionRnd_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en/trainedModels/unionFinetuneRnd/unionRnd_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE_CNTR():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['train_only_readout'] = True
    state['random_readout'] = True
    #state['additional_excludes'] = ['b_0_lm_controller','W_0_lm_controller']
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/unionFinetuneRndCtr/unionRnd_'

    # bleu validation args
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 500
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE_LEAKYALPHA():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False
    state['prefix']='/data/lisatmp3/gulcehrc/nmt/zh-en/trainedModels/unionFineTuneCNT/unionRnd_'

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en/trainedModels/unionFineTuneCNT/unionRnd_valOut.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 500
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE_LEAKYALPHA_arctic():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False
    state['prefix']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/zhenonly_'
    state['target']=["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/binarized_iwslt.en.shuf.h5"]
    state['source']=["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM//binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 64

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True

    return state

def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_rmsprop():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM2/zhenonly_rms_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM2/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM2//binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 32
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['moment'] = 0.98
    state['rho'] = 0.75

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM2/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True

    return state


def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_rmsprop_rho():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rho3/zhenonly_rms_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rho3/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rho3/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 32
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['moment'] = 0.98
    state['rho'] = 0.92

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rho3/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True

    return state

def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_adadelta_rho_half():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rho0.5/zhenonly_adad_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rho0.5/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rho0.5/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 64
    state['algo'] = 'SGD_adadelta'
    state['rho'] = 0.5

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rho0.5/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True

    return state

def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_adadelta_rho_L():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rhoL/zhenonly_adad_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rhoL/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rhoL/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 64
    state['algo'] = 'SGD_adadelta'
    state['rho'] = 0.92

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_rhoL/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True
    return state

def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_adadelta_lm():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr/zhenonly_adad_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 64
    state['algo'] = 'SGD_adadelta'
    state['rho'] = 0.92

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True
    state['use_lm_control'] = True

    return state

def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_adadelta_lm_unbiased():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_unbiased/zhenonly_adad_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_unbiased/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_unbiased/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"

    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'
    state['bs'] = 64
    state['algo'] = 'SGD_adadelta'
    state['rho'] = 0.92
    state['cutoff'] = 5.0
    state['init_ctlr_bias'] = 0

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_unbiased/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0
    state['use_arctic_lm'] = True
    state['use_lm_control'] = True

    return state


def prototype_search_state_with_LM_zh_en_arctic_rmsprop_lm():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/lstm30k_model_reset.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 40
    state['moment'] = 0.96
    state['cutoff'] = 5.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.2

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_arctic_rmsprop_lm():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/lstm30k_model_reset.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 40
    state['moment'] = 0.96
    state['cutoff'] = 5.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.2

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_arctic_rmsprop_lm_fixed():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/lstm30k_model_reset.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 40
    state['moment'] = 0.96
    state['cutoff'] = 5.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.2

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_arctic_rmsprop_lm_fixed_noise():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False
    state['controller_temp'] = 4.0

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed2/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed2/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed2/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/lstm30k_model_reset.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 40
    state['moment'] = 0.96
    state['cutoff'] = 8.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -0.5

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_fixed2/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_arctic_rmsprop_lm_fixed_large():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False
    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_larger/zhenonly_rmslm_large_'
    state['target'] = ['/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_larger/binarized_iwslt.en.shuf.h5']
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_larger/binarized_iwslt.zh.shuf.h5"]
    state['dim'] = 1000
    state['lm_readout_dim'] = 2000
    state['controller_temp'] = 1.0

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_larger/lstm30k_hb_v2_model_ppl145.npz'

    state['bs'] = 128
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 50
    state['moment'] = 0.95
    state['cutoff'] = 6.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_larger/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_arctic_adadelta_lm():
    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_notuning/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_notuning/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_notuning/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/lstm30k_model_reset.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_adadelta'
    state['max_lr_scale'] = 40
    state['moment'] = 0.96
    state['cutoff'] = 5.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.2

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_adad_lmcontr_notuning/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_FINETUNE_arctic_rmsprop_lm():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 40
    state['moment'] = 0.95
    state['cutoff'] = 5.0

    state['use_lm_control'] = True
    state['use_arctic_lm'] = True

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 200
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_with_LM_zh_en_arctic_rmsprop_lm_dbg():

    state = prototype_search_state_with_LM_zh_en_UNION_RAND_FINETUNE()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_dbg/zhenonly_rmslm_'
    state['target'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_dbg/binarized_iwslt.en.shuf.h5"]
    state['source'] = ["/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_dbg/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.zh.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.zh.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/vocab.en.pkl"
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM/lstm30k_finetuned_model.npz'

    state['bs'] = 64
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 40
    state['moment'] = 0.95
    state['cutoff'] = 5.0

    state['use_lm_control'] = True
    state['use_arctic_lm'] = True

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/zh-en_lm/trainedModels/zhenonlyWithArcticLM_rms_lmcontr_notuning_dbg/zhen_only_with_lm.txt'
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_zh_en_without_LM_zhenonly():

    state = prototype_encdec_state()

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False
    state['reload'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 45

    # Source and target sentence
    state['target']=["/data/lisatmp3/gulcehre/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/binarized_iwslt.en.shuf.h5"]
    state['source']=["/data/lisatmp3/gulcehre/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/vocab.en.pkl"

    state['source_encoding'] = 'utf8'
    state['loopIters'] = 3000

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/gulcehre/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/zhenonly_lm_fix_'

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/zhenonly_lm_fix_valOut.txt'
    state['test_set']='/data/lisatmp3/firatorh/nmt/zh-en_lm/tst/IWSLT14.TED.tst2010.zh-en.zh.xml.txt.trimmed'
    state['test_set_grndtruth']='/data/lisatmp3/firatorh/nmt/zh-en_lm/tst/IWSLT14.TED.tst2010.zh-en.en.tok'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_zh_en_without_LM_zhenonly_normalizedBLEU():

    state = prototype_encdec_state()

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False
    state['reload'] = True

    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/binarized_iwslt.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/binarized_iwslt.zh.shuf.h5"]

    # Word -> Id and Id-> Word Dictionaries
    state['indx_word']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/ivocab.en.pkl"
    state['word_indx']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/vocab.zh.pkl"
    state['word_indx_trgt']="/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/vocab.en.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=4839
    state['null_sym_target']=30000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/zhenonly_lm_fix_normBLEU_'

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/zh-en_lm/dev/IWSLT14.TED.dev2010.zh-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/zhenonly_lm_fix_normBLEU_valOut.txt'
    state['test_set']='/data/lisatmp3/firatorh/nmt/zh-en_lm/tst/IWSLT14.TED.tst2010.zh-en.zh.xml.txt.trimmed'
    state['test_set_grndtruth']='/data/lisatmp3/firatorh/nmt/zh-en_lm/tst/IWSLT14.TED.tst2010.zh-en.en.tok'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_zh_en_without_LM_zhenonly_SepNormalizedBLEU():

    state = prototype_search_state_zh_en_without_LM_zhenonly_normalizedBLEU()

    state['prefix']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/zhenonly_lm_fix_sepNormBLEU_'

    # bleu validation args
    state['normalized_bleu'] = True
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/zh-en_lm/trainedModels/zhenonlyWithoutLM/zhenonly_lm_fix_SepNormBLEU_valOut.txt'

    state['dim'] = 500
    state['bs'] = 80
    state['reload'] = False

    state['beam_size'] = 20
    state['bleu_val_frequency'] = 100
    state['validation_burn_in'] = 1000

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_V0():

    state = prototype_search_state()

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/v0_'

    state['reload'] = True
    state['reload_lm'] = False

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    state['include_lm'] = False
    state['train_only_readout'] = False

    state['dim'] = 1200
    state['bs'] = 80
    state['cutoff'] = 1.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/binarized_text.zh.shuf.h5"]
    state['indx_word']       ="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/ivocab.en.pkl"
    state['word_indx']       ="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/vocab.zh.pkl"
    state['word_indx_trgt']  ="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/vocab.en.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=10000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.zh.tok'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/v0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 20000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_TEST():

    state = prototype_search_state_zh_en_without_LM_OPENMT_V0()

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/test_'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/test_valOut.txt'
    state['bleu_val_frequency'] = 1
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_SEGV0():

    state = prototype_search_state_zh_en_without_LM_OPENMT_V0()

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/segV0'

    state['reload'] = False
    state['reload_lm'] = False

    # Source and target sentence
    state['target']=["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en.seg/binarized_text.en.shuf.h5"]
    state['source']=["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en.seg/binarized_text.zh.shuf.h5"]
    state['indx_word']       ="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en.seg/ivocab.zh.pkl"
    state['indx_word_target']="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en.seg/ivocab.en.pkl"
    state['word_indx']       ="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en.seg/vocab.zh.pkl"
    state['word_indx_trgt']  ="/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en.seg/vocab.en.pkl"

    state['source_encoding'] = 'ascii'

    state['null_sym_source']=40000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.zh.tok.seg'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/segV0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 20000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_CTS():

    state = prototype_search_state_zh_en_without_LM_OPENMT_V0()

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_v0_'

    state['reload'] = False
    state['reload_lm'] = False

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05

    state['dim'] = 1200
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30

    state['source_encoding'] = 'utf8'

    state['null_sym_source']=10000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_v0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_CTS_DEEPATTN():

    state = prototype_search_state_zh_en_without_LM_OPENMT_CTS()

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_deepAttn_'
    state['reload'] = True
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_deepAttn_valOut.txt'

    state['deep_attention']= True
    state['deep_attention_n_hids']= [1200,1200]
    state['deep_attention_acts']= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) ']

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_V0_cglr():

    state = prototype_encdec_state()

    state['prefix']='/part/02/Tmp/gulcehrc/nmt/openmt15/trainedModels/withoutLM/adadeltahyper2/v0_adad_'

    state['reload'] = True
    state['reload_lm'] = False
    state['bs'] = 120
    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1200

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.54
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.03

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False
    state['reload'] = True
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/binarized_text.en.shuf.h5"]
    state['source']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/binarized_text.zh.shuf.h5"]
    state['indx_word']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/ivocab.zh.pkl"
    state['indx_word_target']= "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/ivocab.en.pkl"
    state['word_indx']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/vocab.zh.pkl"
    state['word_indx_trgt']  = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/vocab.en.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source'] = 10000
    state['null_sym_target'] = 40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.zh.tok'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.en.tok'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_V0_cglr_sms():

    state = prototype_encdec_state()

    state['prefix']='/part/02/Tmp/gulcehrc/nmt/openmt15/trainedModels/withoutLM/adadeltahyper2_sms/v0_adad_sms_'

    state['reload'] = True
    state['reload_lm'] = False
    state['bs'] = 110
    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1200

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.53
    #0.54
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.02
    #0.03

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False
    state['reload'] = True
    state['saveFreq'] = 30
    state['cutoff'] = 1.0
    state['hookFreq'] = 400

    # Source and target sentence
    state['target']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/binarized_text.en.shuf.h5"]
    state['source']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/binarized_text.zh.shuf.h5"]
    state['indx_word']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/ivocab.zh.pkl"
    state['indx_word_target']= "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/ivocab.en.pkl"
    state['word_indx']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/vocab.zh.pkl"
    state['word_indx_trgt']  = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en/vocab.en.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source'] = 10000
    state['null_sym_target'] = 40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/sms3-dev2.tok.zh'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/sms3-dev2.tok.en'
    state['validation_set_out']='/data/lisatmp3/gulcehrc/nmt/openmt15/trainedModels/withoutLM/v0_valOut_adadelta_sms.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_V0_SGD_adam_seg():

    state = prototype_encdec_state()

    state['prefix']='/part/02/Tmp/gulcehrc/nmt/openmt15/trainedModels/withoutLM/adam_seg/v0_adam_seg_'

    state['reload'] = True
    state['reload_lm'] = False
    state['bs'] = 60
    state['algo'] = 'SGD_adam'
    state['lr'] = 0.00011
    state['dim'] = 1200
    state['cutoff'] = 10.0
    state['hookFreq'] = 400

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.015

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False
    state['reload'] = True
    state['saveFreq'] = 40

    # Source and target sentence
    state['target']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/binarized_text.en.shuf.h5"]
    state['source']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/binarized_text.zh.shuf.h5"]
    state['indx_word']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/ivocab.zh.pkl"
    state['indx_word_target']= "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/ivocab.en.pkl"
    state['word_indx']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/vocab.zh.pkl"
    state['word_indx_trgt']  = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/vocab.en.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source'] = 40000
    state['null_sym_target'] = 40000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 15

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] ='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.zh.tok.seg'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/val.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/gulcehrc/nmt/openmt15/trainedModels/withoutLM/v0_valOut_adam_seg.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 20000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_V0_SGD_adadelta_seg():

    state = prototype_encdec_state()

    state['prefix']='/part/02/Tmp/gulcehrc/nmt/openmt15/trainedModels/withoutLM/adadelta_seg/v0_adadelta_seg_'

    state['reload'] = True
    state['reload_lm'] = False
    state['bs'] = 80
    state['algo'] = 'SGD_adadelta'
    state['dim'] = 1200
    state['cutoff'] = 1.0
    state['hookFreq'] = 400

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.015

    state['include_lm'] = False
    state['reload_lm'] = False
    state['train_only_readout'] = False
    state['reload'] = True
    state['saveFreq'] = 30

    # Source and target sentence
    state['target']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/binarized_text.en.shuf.h5"]
    state['source']          = ["/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/binarized_text.zh.shuf.h5"]
    state['indx_word']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/ivocab.zh.pkl"
    state['indx_word_target']= "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/ivocab.en.pkl"
    state['word_indx']       = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/vocab.zh.pkl"
    state['word_indx_trgt']  = "/part/02/Tmp/gulcehrc/nmt/openmt15/bitext.zh-en.seg/vocab.en.pkl"

    state['source_encoding'] = 'utf8'

    state['null_sym_source'] = 65000
    state['null_sym_target'] = 35000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/sms3-dev2.tok.zh.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/sms3-dev2.tok.en'
    state['validation_set_out'] = '/data/lisatmp3/gulcehrc/nmt/openmt15/trainedModels/withoutLM/v0_valOut_adadelta_sms.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 10000

    return state

def prototype_search_state_zh_en_without_LM_OPENMT_CTS_DEEPDEC():

    state = prototype_search_state_zh_en_without_LM_OPENMT_CTS()

    state['dim'] = 1000
    state['decoder_stack'] = 2
    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_deepDec_'
    state['reload'] = True
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_deepDec_valOut.txt'
    state['weight_noise_amount'] = 0.001
    state['dropout'] = 0.55

    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 30000
    return state

def prototype_search_state_with_LM_zh_openmt_rmsprop_deep_fusion_cts():
    state = prototype_search_state_zh_en_without_LM_OPENMT_V0_cglr_sms()

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/openmt15/trainedModels/rmsprop_withlm_cts/v0_rmsprop_cts_'
    state['controller_temp'] = 1.0

    # Word -> Id and Id-> Word Dictionaries
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/openmt15/caglar/lm_model/lstm40k_hb_v2_model.npz'
    state['dim_lm'] = 2400
    state['dim'] = 1200

    state['bs'] = 128
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 50
    state['moment'] = 0.96
    state['cutoff'] = 6.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0

    # bleu validation args
    state['validation_set_out'] = '/data/lisatmp3/gulcehrc/nmt/openmt15/trainedModels/rmsprop_withlm_cts/zhen_only_with_lm_cts.txt'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'

    state['verbose'] = True
    state['beam_size'] = 20

    # bleu validation score
    state['bleu_val_frequency'] = 500
    state['validation_burn_in'] = 0

    return state


def prototype_double_state():
    state = prototype_encdec_state()
    state['target'] = ["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/binarized_text.zh.shuf.h5"]
    state['source'] = ["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/binarized_text.en.shuf.h5"]
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/ivocab.zh.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/vocab.zh.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/vocab.en.pkl"
    state['utf8'] = True
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['enc_rec_layer_hier'] = 'DoubleRecurrentLayer'
    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix'] = '/data/lisatmp3/gulcehrc/nmt/openmt15/trainedModels/rmsprop_nolm_doublehiear/double_hiear_'

    state['bs'] = 128
    state['algo'] = 'SGD_rmsprop'
    state['lr'] = 1e-4
    state['max_lr_scale'] = 50
    state['moment'] = 0.96
    state['cutoff'] = 6.0

    state['use_noise'] = True
    state['dropout'] = 0.54
    #0.54
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05
    #0.03

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 1000
    state['validation_burn_in'] = 1000
    state['use_lm_control'] = False
    state['use_arctic_lm'] = False


    state['null_sym_source'] = 4000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1
    state['seqlen'] = 30
    state['bs'] = 80
    state['dim'] = 1000
    state['rank_n_approx'] = 256
    return state

def prototype_double_state_cts():
    state = prototype_search_state_zh_en_without_LM_OPENMT_V0_cglr_sms()
    state['enc_rec_layer'] = 'DoubleRecurrentLayer'
    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/hier_cts_v0_'

    state['bs'] = 128
    state['algo'] = 'SGD_adadelta'
    state['cutoff'] = 5.0

    state['use_noise'] = True
    state['dropout'] = 0.54
    #0.54
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05
    #0.03

    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/hier_cts_v0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 20000

    state['use_lm_control'] = False
    state['use_arctic_lm'] = False

    state['null_sym_source'] = 10000
    state['null_sym_target'] = 40000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1
    state['seqlen'] = 40
    state['bs'] = 80
    state['dim'] = 1000
    state['rank_n_approx'] = 620

    return state

def prototype_double_state_cts_small():
    state = prototype_double_state_cts()
    state['enc_rec_layer'] = 'DoubleRecurrentLayer'
    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['sort_k_batches'] = 20
    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/hier_cts_v0_'

    state['bs'] = 128
    state['algo'] = 'SGD_adadelta'
    state['cutoff'] = 5.0

    state['use_noise'] = True
    state['dropout'] = 0.54
    #0.54
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05
    #0.03

    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/hier_cts_v0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 20000

    state['use_lm_control'] = False
    state['use_arctic_lm'] = False

    state['null_sym_source'] = 10000
    state['null_sym_target'] = 40000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1
    state['seqlen'] = 40
    state['bs'] = 80
    state['dim'] = 1000
    state['rank_n_approx'] = 620

    return state


def prototype_bidir_hier_cts():
    state = prototype_search_state_zh_en_without_LM_OPENMT_V0_cglr_sms()

    state['reload'] = True

    state['source'] = ["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/binarized_text.zh.shuf.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/binarized_text.en.shuf.h5"]

    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/ivocab.zh.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/vocab.zh.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/openmt15/bitext.zh-en/vocab.en.pkl"
    state['utf8'] = True

    state['enc_rec_layer'] = 'DoubleRecurrentLayer'
    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'

    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = False
    state['use_hier_enc'] = True
    state['reload_lm'] = False
    state['source_encoding'] = 'utf8'
    state['source_splitted'] = True

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/double_hier_adadelta_'

    state['bs'] = 80
    state['algo'] = 'SGD_adadelta'
    state['cutoff'] = 6.0

    state['use_noise'] = True
    state['dropout'] = 0.5

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    state['null_sym_source'] = 5000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/double_hier_adadelta_out.txt'
    state['validation_set']= '/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000
    state['use_lm_control'] = False
    state['use_arctic_lm'] = False
    state['hookFreq'] = 30

    state['dim'] = 750
    state['rank_n_approx'] = 256

    return state

def prototype_bidir_hier_cts_adam():
    state = prototype_bidir_hier_cts()

    state['reload'] = True

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/double_hier_adam_'

    state['bs'] = 80
    state['algo'] = 'SGD_adam'
    state['lr'] = 0.0001
    state['cutoff'] = 10.0

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['optimize_probs'] = True
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/double_hier_adam_out.txt'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000

    return state


def prototype_search_state_zh_en_without_LM_OPENMT_CTS_DBG():

    state = prototype_search_state_zh_en_without_LM_OPENMT_V0()

    state['prefix']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_v0_dbg_'

    state['reload'] = True
    state['reload_lm'] = False

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05

    state['dim'] = 1200
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['hookFreq'] =400
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 5

    state['source_encoding'] = ''

    state['null_sym_source']=10000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.zh'
    state['validation_set_grndtruth']='/data/lisatmp3/firatorh/nmt/openmt15/dev.zh-en.raw1/newDevTst/p3ctstune2.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/openmt15/trainedModels/withoutLM/cts_v0_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 0

    return state


def prototype_search_state_tr_en_without_LM_ACL_ADADELTA():

    state = prototype_search_state()

    state['source'] =          ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.tr.shuf.h5"]
    state['target'] =          ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.en.shuf.h5"]
    state['indx_word'] =        "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.tr.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.en.pkl"
    state['word_indx'] =        "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.tr.pkl"
    state['word_indx_trgt'] =   "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.en.pkl"
    state['prefix']='/data/lisatmp3/firatorh/nmt/acl15/trainedModels/withoutLM/v0_adadelta_'

    state['reload'] = True
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05

    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    state['null_sym_source']=30000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/acl15/trainedModels/withoutLM/v0_adadelta_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_tr_en_without_LM_ACL_ADADELTA_TEST():

    state = prototype_search_state()

    state['source'] =          ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.tr.shuf.h5"]
    state['target'] =          ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.en.shuf.h5"]
    state['indx_word'] =        "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.tr.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.en.pkl"
    state['word_indx'] =        "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.tr.pkl"
    state['word_indx_trgt'] =   "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.en.pkl"
    state['prefix']='/data/lisatmp3/firatorh/nmt/acl15/trainedModels/withoutLM/v0_adadelta_test_'

    state['reload'] = False
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05

    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    state['null_sym_source']=30000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = None #'/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set']='/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.en.tok'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/acl15/trainedModels/withoutLM/v0_adadelta_valOut.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000

    return state


def prototype_search_state_tr_en_withLM_iwslt_adam_noController():

    state = prototype_encdec_state()
    state['prefix']='/data/lisatmp3/firatorh/nmt/acl15/trainedModels/deep_fusion_wc/adam_iwslt_withLM_noController/v0_adam_'

    state['reload'] = True
    state['reload_lm'] = False
    state['bs'] = 128

    state['algo'] = 'SGD_adam'
    state['lr'] = 2*1e-4

    #state['algo'] = 'SGD_adadelta'
    #state['lr'] = 1e-6
    #state['dim'] = 1200
    #state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = False
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/en_gigaword_adam/lstm40k_hb_v2_model.npz'

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05

    state['validFreq'] = 2000

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 400

    # Source and target sentence
    state['target']          = ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.en.shuf.h5"]
    state['source']          = ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.tr.shuf.h5"]
    state['indx_word']       = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.tr.pkl"
    state['indx_word_target']= "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.en.pkl"
    state['word_indx']       = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.tr.pkl"
    state['word_indx_trgt']  = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.en.pkl"

    state['source_encoding'] = 'ascii'

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 40000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/deep_fusion_wc/adam_iwslt_withLM_noController/val_outset_adam.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 500
    state['validation_burn_in'] = 1000

    return state

def prototype_search_state_tr_en_withLM_iwslt_adam_TMController():

    state = prototype_encdec_state()
    state['prefix']='/data/lisatmp3/firatorh/nmt/acl15/trainedModels/deep_fusion_wc/adam_iwslt_withLM_noController/v0_adam_'

    state['reload'] = True
    state['reload_lm'] = False
    state['bs'] = 128

    state['algo'] = 'SGD_adam'
    state['lr'] = 2*1e-4

    #state['algo'] = 'SGD_adadelta'
    #state['lr'] = 1e-6
    #state['dim'] = 1200
    #state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0

    state['source_splitted'] = True

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/en_gigaword_adam/lstm40k_hb_v2_model.npz'

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05

    state['validFreq'] = 2000

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 400

    # Source and target sentence
    state['target']          = ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.en.shuf.h5"]
    state['source']          = ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.tr.shuf.h5"]
    state['indx_word']       = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.tr.pkl"
    state['indx_word_target']= "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.en.pkl"
    state['word_indx']       = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.tr.pkl"
    state['word_indx_trgt']  = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.en.pkl"

    state['source_encoding'] = 'ascii'

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 40000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False

    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/deep_fusion_wc/adam_iwslt_withLM_noController/val_outset_adam.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 500
    state['validation_burn_in'] = 1000

    return state

def prototype_search_state_fi_en_without_LM_WMT_ADADELTA():

    state = prototype_search_state()

    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.shuf.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.shuf.en.h5"]
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.fi.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.fi.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.en.pkl"
    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_'

    state['reload'] = True
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01
    state['optimize_probs'] = False

    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 40
    state['sort_k_batches'] = 20

    state['null_sym_source']=50000
    state['null_sym_target']=50000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/refGHOG_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_fi_en_without_LM_WMT_ADAM_40k():

    state = prototype_search_state_fi_en_without_LM_WMT_ADADELTA()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adam_40k_'
    state['reload'] = False
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.015
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['algo'] = 'SGD_adam'
    state['lr'] = 0.00011
    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['cutoff'] = 10.0
    state['bs'] = 80

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    state['null_sym_source']=40000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/refGHOG_adam_40k_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_fi_en_without_LM_WMT_ADAM_40k_CONT():

    state = prototype_search_state_fi_en_without_LM_WMT_ADADELTA()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adam_40k_cont_adadelta_'
    state['reload'] = True
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['algo'] = 'SGD_adadelta'
    #state['lr'] = 0.0001
    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['cutoff'] = 10.0
    state['bs'] = 80

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    state['null_sym_source']=40000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/refGHOG_adam_40k_cont_adadelta_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_fi_en_without_LM_WMT_ADADELTA_40k():

    state = prototype_search_state_fi_en_without_LM_WMT_ADADELTA()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_'

    state['reload'] = False
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.015
    state['optimize_probs'] = False

    state['enc_rec_layer'] = 'RecurrentLayer'
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    state['null_sym_source']=40000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/refGHOG_adadelta_40k_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_tr_en_withoutLM_fast_hierV0():

    state = prototype_encdec_state()
    state['prefix'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/fast_hierV0_model.npz'

    state['reload'] = False
    state['reload_lm'] = False

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['bs'] = 128

    state['include_lm'] = False
    state['reload_lm'] = False
    state['reload'] = False

    state['use_noise'] = True
    state['dropout'] = 0.5

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 400

    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # Source and target sentence
    state['target']          = ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.en.shuf.h5"]
    state['source']          = ["/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/binarized_text.tr.shuf.h5"]
    state['indx_word']       = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.tr.pkl"
    state['indx_word_target']= "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/ivocab.en.pkl"
    state['word_indx']       = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.tr.pkl"
    state['word_indx_trgt']  = "/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/vocab.en.pkl"

    state['source_encoding'] = 'ascii'
    state['source_splitted'] = True

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 40000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    # Hierarchical attention parameters
    state['use_hier_enc'] = True
    state['enc_rec_layer'] = 'DoubleRecurrentLayer'
    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = False

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.tr.tok.seg'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/acl15/bitext.tr-en/IWSLT14.TED.dev.tst2010.tr-en.en.tok'
    state['validation_set_out'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/fast_hier_outset.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_tr_en_withoutLM_fast_hier_TEST_BETAS():

    state = prototype_search_state_tr_en_withoutLM_fast_hierV0()
    state['prefix'] = '/data/lisatmp3/firatorh/nmt/acl15/trainedModels/fast_hier_TEST_BETA_'
    '''
    state['use_hier_enc'] = False
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    '''
    state['reload'] = False
    state['reload_lm'] = False
    state['hookFreq'] = 5

    return state

def prototype_search_state_DEBUG():

    state = prototype_search_state()

    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.shuf.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.shuf.en.h5"]
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.fi.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.fi.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.en.pkl"
    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/TMP_DEBUG_'

    state['reload'] = False
    state['reload_lm'] = False

    state['use_noise'] = False
    state['optimize_probs'] = False

    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 40
    state['sort_k_batches'] = 20

    state['null_sym_source']=3000
    state['null_sym_target']=3000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = None

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController():

    state = prototype_encdec_state()
    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_'


    #state['algo'] = 'SGD_adam'
    #state['lr'] = 2*1e-4

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000
    state['bs'] = 80

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 30

    # Source and target sentence
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.shuf.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.shuf.en.h5"]
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.fi.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.fi.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.en.pkl"

    state['null_sym_source'] = 40000
    state['null_sym_target'] = 40000

    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_out.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController_cont():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_cont_'

    state['algo'] = 'SGD_adadelta'
    state['bs'] = 120

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01
    state['optimize_probs'] = False

    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_cont_out.txt'

    state['bleu_val_frequency'] = 500
    state['validation_burn_in'] = 1000

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController_resetReadout():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_resetReadout_'

    state['algo'] = 'SGD_adadelta'
    state['bs'] = 80

    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.fi.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab_ghog.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.fi.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab_ghog.en.pkl"

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = 0.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = True
    state['dropout'] = 0.55
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 30

    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_resetReadout_out.txt'

    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 20000

    return state


def prototype_search_state_cs_en_40k_WMT_ADADELTA():

    state = prototype_search_state()

    datadir = '/data/lisatmp3/jeasebas/nmt/data/wmt15/cs-en/tok.apos.clean.shuf/'
    vocabdir = '/data/lisatmp3/firatorh/nmt/wmt15/data/cs-en/'
    state['source'] = [datadir + "all.tok.apos.clean.shuf.cs-en.cs.h5"]
    state['target'] = [datadir + "all.tok.apos.clean.shuf.cs-en.en.h5"]
    state['indx_word'] = vocabdir + "ivocab.cs-en.cs.pkl"
    state['indx_word_target'] = vocabdir + "ivocab.cs-en.en.pkl"
    state['word_indx'] = vocabdir + "vocab.cs-en.cs.pkl"
    state['word_indx_trgt'] = vocabdir + "vocab.cs-en.en.pkl"
    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_cs_en_'

    state['reload'] = False
    state['reload_lm'] = False

    state['use_noise'] = False
    state['dropout'] = 1.0
    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.00
    state['optimize_probs'] = False

    state['dim'] = 1000
    state['rank_n_approx'] = 620
    state['bs'] = 80
    state['cutoff'] = 5.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['search'] = True
    state['last_forward'] = False
    state['last_backward'] = False
    state['forward'] = True
    state['backward'] = True
    state['seqlen'] = 50
    state['sort_k_batches'] = 12

    state['null_sym_source'] = 40000
    state['null_sym_target'] = 40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['normalized_bleu'] = True
    state['bleu_script'] = None #'/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/refGHOG_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 10000
    state['validation_burn_in'] = 20000

    return state

def prototype_search_state_fi_en_without_LM_WMT_ADADELTA_40k_RESHUF():

    state = prototype_search_state_fi_en_without_LM_WMT_ADADELTA()

    state['seed'] = 4603

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_reshuf_'
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.fi.reshuf.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.en.reshuf.h5"]
    state['indx_word'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.fi.pkl"
    state['indx_word_target'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/ivocab.en.pkl"
    state['word_indx'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.fi.pkl"
    state['word_indx_trgt'] = "/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/vocab.en.pkl"

    state['reload'] = True
    state['reload_lm'] = False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01
    state['optimize_probs'] = False

    state['enc_rec_layer'] = 'RecurrentLayer'
    state['bs'] = 80
    state['cutoff'] = 10.0
    state['algo'] = 'SGD_adadelta'
    state['saveFreq'] = 30
    state['hookFreq'] = 30

    state['seqlen'] = 50
    state['sort_k_batches'] = 20

    state['null_sym_source']=40000
    state['null_sym_target']=40000

    state['n_sym_source']=state['null_sym_source'] + 1
    state['n_sym_target']=state['null_sym_target'] + 1

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set_out']= '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/refGHOG_adadelta_40k_reshuf_out.txt'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.seg.fi'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.en'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 5000
    state['validation_burn_in'] = 75000

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController_reshuf0():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_'

    state['seed'] = 4321

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000
    state['bs'] = 80

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = 1
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = True
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.05
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['cutoff'] = 10.0
    state['hookFreq'] = 30

    # Source and target sentence
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf0.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf0.en.h5"]

    # bleu validation args
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_out.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 5000

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController_reshuf1():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf1_'

    state['seed'] = 5413

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000
    state['bs'] = 80

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = 1
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.0
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 30

    # Source and target sentence
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf1.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf1.en.h5"]

    # bleu validation args
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_2.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_2.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf1_out.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_TMController_reshuf0():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_tmCont_'

    state['seed'] = 1321

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000
    state['bs'] = 80

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = 2
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = True
    state['dropout'] = 0.5
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.0
    state['optimize_probs'] = False

    state['saveFreq'] = 30
    state['cutoff'] = 5.0
    state['hookFreq'] = 30

    # Source and target sentence
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf0.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf0.en.h5"]

    # bleu validation args
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.seg.fi'
    state['validation_set_grndtruth'] ='/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.en'
    state['validation_set_out']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_tmCont_out.txt'

    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_vecLMController_reshuf1_noNoise():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_'

    state['seed'] = 9911

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000
    state['bs'] = 80

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.5
    state['use_lm_control'] = 1
    state['vector_controller'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = -1.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False

    state['use_noise'] = False
    state['dropout'] = 1.0
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.0
    state['optimize_probs'] = False

    state['saveFreq'] = 45
    state['cutoff'] = 5.0
    state['hookFreq'] = 30

    # Source and target sentence
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf1.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf1.en.h5"]

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_2.tok.seg.fi'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_2.tok.en'
    state['validation_set_out']= '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf1_noNoise_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 100000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_vecLMController_reshuf1_noNoise_TEST():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_LMController()

    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_TEST_'

    state['seed'] = 111

    state['algo'] = 'SGD_adadelta'
    state['lr'] = 1e-6
    state['dim'] = 1000
    state['dim_lm'] = 2000
    state['lm_readout_dim'] = 2000
    state['bs'] = 80

    state['include_lm'] = True
    state['reload_lm'] = True
    state['reload'] = True
    state['train_only_readout'] = True
    state['random_readout'] = False

    state['controller_temp'] = 1.0
    state['use_lm_control'] = 1
    state['vector_controller'] = True
    state['use_arctic_lm'] = True
    state['init_ctlr_bias'] = 0.0
    state['rho'] = 0.0 # this is only used if use_lm_control=False
    state['additional_ngrad_monitors'] = ['W_0_lm_controller', 'W_0_dec_lm_embed_0']

    state['use_noise'] = False
    state['dropout'] = 1.0
    state['modelpath'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/lstm40k_intersect_ppl78_model.npz'

    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.0
    state['optimize_probs'] = False

    state['saveFreq'] = 45
    state['cutoff'] = 5.0
    state['hookFreq'] = 30

    # Source and target sentence
    state['source'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf1.fi.h5"]
    state['target'] = ["/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/binarized_text.reshuf1.en.h5"]

    # bleu validation args
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['validation_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    state['validation_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    state['validation_set_out']= '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf1_noNoise_TEST_out.txt'
    state['output_validation_set'] = True
    state['beam_size'] = 20
    state['bleu_val_frequency'] = 2000
    state['validation_burn_in'] = 0

    return state

def prototype_search_state_fi_en_withLM_gigaword_adadelta_vecLMController_reshuf1_noNoise_TEST2():

    state = prototype_search_state_fi_en_withLM_gigaword_adadelta_vecLMController_reshuf1_noNoise_TEST()
    state['use_cross_dict'] = True
    state['additional_ngrad_monitors'] = ['W_0_lm_controller', 'W_0_dec_lm_embed_0']
    state['prefix']='/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_TEST2_'
    return state

