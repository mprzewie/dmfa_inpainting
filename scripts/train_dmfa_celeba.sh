#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=3


ARGS="--dataset celeba --img_size 64 --mask_train_size 32 --mask_val_size 32 --architecture fullconv \
    --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 4 --num_factors 4 --a_amplitude 0.2 --batch_size=32 --dump_sample_results --render_every=3"

# EXPERIMENT_NAME="fullconv/64x64/dmfa_mse_10_eps_v4"

# python train_inpainter.py \
#     --experiment_name $EXPERIMENT_NAME --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --lr=4e-5 $ARGS 

# python train_inpainter.py \
#     --experiment_name $EXPERIMENT_NAME --num_epochs 40 --l_nll_weight 1 --l_mse_weight 0 --lr=1e-5 $ARGS 

EXPERIMENT_NAME="fullconv/64x64/incomplete_data/dmfa_mse_10_eps_v5_train_det" # TODO


python train_inpainter.py \
    --experiment_name $EXPERIMENT_NAME --num_epochs 7 --l_nll_weight 1 --l_mse_weight 1 --lr=4e-5 $ARGS --mask_unknown_size 32

python train_inpainter.py \
    --experiment_name $EXPERIMENT_NAME --num_epochs 40 --l_nll_weight 1 --l_mse_weight 0 --lr=1e-5 $ARGS --mask_unknown_size 32 


# context encoder

# ARGS="--dataset celeba --img_size 64 --mask_train_size 32 --mask_val_size 32 --architecture fullconv \
#     --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 4 --num_factors 1 --a_amplitude 0.2 --batch_size=32 --dump_sample_results --render_every=3"

# EXPERIMENT_NAME="fullconv/64x64/context_encoder_v1" 

# python train_inpainter.py \
#     --experiment_name $EXPERIMENT_NAME --num_epochs 50 --l_nll_weight 0 --l_mse_weight 1 --lr=4e-5 $ARGS 

# EXPERIMENT_NAME="fullconv/64x64/incomplete_data/context_encoder_v1" 

# python train_inpainter.py \
#     --experiment_name $EXPERIMENT_NAME --num_epochs 50 --l_nll_weight 0 --l_mse_weight 1 --lr=4e-5 $ARGS --mask_unknown_size 32