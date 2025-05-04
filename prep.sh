#!/bin/bash

# prepare meta info and trials for voxceleb
python -m utils.prep_vox_meta
python -m utils.prep_vox_train
python -m utils.prep_vox_trials

# prepare meta info and trials for cnceleb
#python -m utils.prep_cn_meta
#python -m utils.prep_cn_train
#python -m utils.prep_cn_trials
