#!/bin/bash -x

THEANO_FLAGS="floatX=float32,device=gpu,force_device=True" python -m ipdb train.py --proto=$1
