#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=3


EXPERIMENT_NAME="fullconv/64x64/incomplete_data/dmfa_mse_10_eps_v5_train_det"


python train_inpainter.py \
    --experiment_name $EXPERIMENT_NAME --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --lr=4e-5 $ARGS --mask_unknown_size 32

python train_inpainter.py \
    --experiment_name $EXPERIMENT_NAME --num_epochs 40 --l_nll_weight 1 --l_mse_weight 0 --lr=1e-5 $ARGS --mask_unknown_size 32 
