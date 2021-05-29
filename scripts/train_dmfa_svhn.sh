#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=2

# EXPERIMENT_NAME=fullconv/complete_data/dmfa_mse_10_eps_v4_train_det

# ARGS="--experiment_name ${EXPERIMENT_NAME} --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 \
#     --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2"

# # complete data
# python train_inpainter.py  $ARGS  --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --lr 4e-4
# python train_inpainter.py $ARGS --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0 --lr 1e-4

# incomplete data
EXPERIMENT_NAME=fullconv/incomplete_data/dmfa_mse_10_eps_v4_train_det

ARGS="--experiment_name ${EXPERIMENT_NAME} --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 \
    --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --mask_unknown_size 16"

python train_inpainter.py  $ARGS  --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --lr 4e-4
python train_inpainter.py $ARGS --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0 --lr 1e-4


# EXPERIMENT_NAME=fullconv/complete_data/context_encoder_v1

# ARGS="--experiment_name ${EXPERIMENT_NAME} --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 \
#     --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 1 --a_amplitude 0.2"

# python train_inpainter.py  $ARGS  --num_epochs 100 --l_nll_weight 0 --l_mse_weight 1 --lr 4e-4


# EXPERIMENT_NAME=fullconv/incomplete_data/context_encoder_v1

# ARGS="--experiment_name ${EXPERIMENT_NAME} --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 \
#     --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 1 --a_amplitude 0.2 --mask_unknown_size 16"

# python train_inpainter.py  $ARGS  --num_epochs 100 --l_nll_weight 0 --l_mse_weight 1 --lr 4e-4
