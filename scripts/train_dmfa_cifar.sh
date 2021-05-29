#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=1

EXPERIMENT_NAME=fullconv/complete_data/dmfa_mse_10_eps_v2_ctd

ARGS=" --dataset cifar10 --img_size 32 --mask_train_size 16 --mask_val_size 16 \
    --batch_size 128 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2"

 complete data
 python train_inpainter.py  $ARGS  --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --experiment_name ${EXPERIMENT_NAME}
 python train_inpainter.py $ARGS --num_epochs 50 --l_nll_weight 1 --l_mse_weight 0 --experiment_name ${EXPERIMENT_NAME}

# incomplete data
EXPERIMENT_NAME=fullconv/incomplete_data/dmfa_mse_10_eps_v3_train_det


python train_inpainter.py  $ARGS  --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --mask_unknown_size 16 --experiment_name ${EXPERIMENT_NAME}
python train_inpainter.py $ARGS --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0  --mask_unknown_size 16 --experiment_name ${EXPERIMENT_NAME}