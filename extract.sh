#!/bin/bash


declare -a ckpt_times=('20240514_1701')

for ckpt_time in "${ckpt_times[@]}"; do
    for ((ckpt_num=40; ckpt_num<=40; ckpt_num+=1)); do
        python extract.py --task=voxceleb1 --ckpt_dir=model_ckpt/ckpt_${ckpt_time} --ckpt_num=${ckpt_num} --n_jobs=1 --job=0
    done
done
