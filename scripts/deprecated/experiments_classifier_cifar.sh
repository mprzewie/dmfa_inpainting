#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=2

EXPERIMENTS_DIR=exp_v1

DEFAULT_ARGS="--dataset cifar10  --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=10 --lr=1e-4 \
        --dump_sample_results --render_every 3 --max_benchmark_batches=-1 \
        --cls_bl 3 --cls_latent_size 256 --cls_dropout=0.5 --cls_depth 3"


# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_gt --convar_type=naive --inpainter_type=gt $DEFAULT_ARGS 
    
# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_zero --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS 

# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_noise --convar_type=naive --inpainter_type=noise $DEFAULT_ARGS 


# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_fc_comp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/cifar10/fullconv/complete_data/dmfa_mse_10_eps_v2/ $DEFAULT_ARGS 

# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_fc_comp_cnv \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/cifar10/fullconv/complete_data/dmfa_mse_10_eps_v2/ $DEFAULT_ARGS \
#     --convar_type naive

# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_fc_incomp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/cifar10/fullconv/incomplete_data/dmfa_mse_10_eps_v2/ $DEFAULT_ARGS 

# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_fc_incomp_cnv \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/cifar10/fullconv/incomplete_data/dmfa_mse_10_eps_v2/ $DEFAULT_ARGS \
#     --convar_type naive

conda activate acflow



# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/mfa_frozen \
#     --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/cifar10_32_32/ $DEFAULT_ARGS

# python train_classifier.py \
#     --experiment_name=$EXPERIMENTS_DIR/mfa_frozen_cnv \
#     --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/cifar10_32_32/ $DEFAULT_ARGS \
#     --convar_type naive

python train_classifier.py --experiment_name=$EXPERIMENTS_DIR/inp_acflow \
    --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/cifar/rnvp/ $DEFAULT_ARGS

