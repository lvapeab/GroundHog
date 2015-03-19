import numpy as np
import cPickle

if __name__ == '__main__':
    models_path = '/data/lisatmp3/firatorh/turkishParallelCorpora/compiled/tr-en/trainedModels/'

    model_filename = 'transferIWSLT14_model.npz'
    state_filename = 'transferIWSLT14_state.pkl'
    bleuscore_file = 'transferIWSLT14_val_bleu_scores.npz'


    model = np.load(models_path + model_filename)
    state = {}
    with open(models_path + state_filename) as src:
        state.update(cPickle.load(src))

    scores = model = np.load(models_path + bleuscore_file)

    bleu_scores = model['bleu_scores']
    
    for i in bleu_scores:
        print(str(i) + ' ')