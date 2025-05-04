#!/bin/bash

# for voxceleb
declare -a ckpt_times=('20240514_1701')

for ckpt_time in "${ckpt_times[@]}"; do
    for ((ckpt_num=40; ckpt_num<=40; ckpt_num+=1)); do
         python -m be_run.be_voxceleb --ckpt_num=${ckpt_num} --ckpt_time=${ckpt_time} --is_snorm
    done
done

exit

# for cnceleb
#declare -a ckpt_times=('20240514_1701')
#
#for ckpt_time in "${ckpt_times[@]}"; do
#    for ((ckpt_num=40; ckpt_num<=40; ckpt_num+=1)); do
#         python -m be_run.be_cnceleb --ckpt_num=${ckpt_num} --ckpt_time=${ckpt_time} --is_snorm
#    done
#done
