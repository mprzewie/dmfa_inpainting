#!/usr/bin/env bash

source ~/.bashrc
conda activate rapids

set -xe 

export CUDA_VISIBLE_DEVICES=1

EXPERIMENTS_DIR=64x64/experiments_v5_long_training

# v2
DEFAULT_ARGS="--batch_size 24 --dataset celeba  --img_size 64 --mask_train_size 0 --mask_unknown_size 32 --mask_val_size 32 --num_epochs=50 --render_every 2 --lr=1e-4 \
     --wae_fc=64 --wae_lc=64 --wae_depth=4 --wae_bl=4 --wae_latent_size=64 --wae_disc_hidden 512 --wae_recon_loss mse \
     --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --max_benchmark_batches 1500 --wae_disc_loss_weight 0.025"

{
# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_gt --convar_type=naive --inpainter_type=gt $DEFAULT_ARGS


# python train_wae_v2.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_zero_appnd --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS --convar_append_mask

# python train_wae_v2.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_zero --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS

# python train_wae_v2.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp_v2 \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/celeba/fullconv/64x64/incomplete_data/dmfa_mse_10_eps_v5_train_det $DEFAULT_ARGS


# python train_wae_v2.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp_v2 \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/celeba/fullconv/64x64/incomplete_data/dmfa_mse_10_eps_v5_train_det $DEFAULT_ARGS

# python train_wae_v2.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp_cnv_v2 \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/celeba/fullconv/64x64/incomplete_data/dmfa_mse_10_eps_v5_train_det $DEFAULT_ARGS \
#     --convar_type naive


# conda activate acflow

# python train_wae.py --experiment_name=$EXPERIMENTS_DIR/inp_acflow \
#     --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/celeba/rnvp/ $DEFAULT_ARGS 

# conda activate uj

# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/partial --convar_type=partial --inpainter_type=zero $DEFAULT_ARGS 

# conda activate rapids 
# python train_wae_v2.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_knn_5k --convar_type=naive --inpainter_type=knn $DEFAULT_ARGS

# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_noise --convar_type=naive --inpainter_type=noise $DEFAULT_ARGS


# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/dmfa_comp_cnv \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/celeba/fullconv/64x64/dmfa_mse_10_eps_v4 $DEFAULT_ARGS \
#     --convar_type naive

python train_wae_v2.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/celeba_64_64/ $DEFAULT_ARGS
exit
}