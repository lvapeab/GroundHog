#!/bin/bash


trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py

state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/europarl/stateEuroparl620_1000.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
# prototype="--proto=prototype_search_state_with_LM_tr_en"

#skipinit=""
skipinit="--skip-init"


python $trainer --state=$state ${skipinint} ${prototype}