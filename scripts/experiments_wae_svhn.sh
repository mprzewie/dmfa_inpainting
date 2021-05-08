#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 

export CUDA_VISIBLE_DEVICES=2

# EXPERIMENTS_DIR=ae_v3_long_training

# DEFAULT_ARGS="--batch_size 64 --dataset svhn  --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --render_every 2 \
#                 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --lr 2e-4 \
#                 --wae_disc_loss_weight 0 --skip_fid \
#                 --wae_recon_loss mse --wae_bl 2 --wae_depth 2 --max_benchmark_batches -1 --wae_latent_size 60 --wae_fc 32 --wae_lc 96"

EXPERIMENTS_DIR=wae_v10_long_training
DEFAULT_ARGS="--batch_size 64 --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --render_every 5 \
                --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --lr 4e-4 --wae_recon_loss mse --wae_bl 2 --wae_depth 2 \
                --max_benchmark_batches -1 --wae_latent_size 20 --wae_fc 96 --wae_lc 96 --wae_disc_loss_weight 0.01"

{
# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_gt --convar_type=naive --inpainter_type=gt $DEFAULT_ARGS

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_zero --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_noise --convar_type=naive --inpainter_type=noise $DEFAULT_ARGS

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_zero_appnd --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS --convar_append_mask

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_dmfa_fc_comp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/dmfa_mse_10_eps/ $DEFAULT_ARGS

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_dmfa_fc_comp_cnv \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/dmfa_mse_10_eps/ $DEFAULT_ARGS --convar_type naive

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_dmfa_fc_incomp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/dmfa_mse_10_eps/ $DEFAULT_ARGS 

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_dmfa_fc_incomp_cnv \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/dmfa_mse_10_eps/ $DEFAULT_ARGS --convar_type naive

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/ce_comp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/context_encoder_v1/ $DEFAULT_ARGS --convar_type naive

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/ce_incomp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/context_encoder_v1/ $DEFAULT_ARGS --convar_type naive

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/partial_conv --convar_type=partial --inpainter_type=zero $DEFAULT_ARGS 

conda activate acflow

python train_wae.py --experiment_name=$EXPERIMENTS_DIR/inp_acflow \
    --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/svhn/rnvp/ $DEFAULT_ARGS 

conda activate uj

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/svhn_32_32/ $DEFAULT_ARGS 

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa_cnv \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/svhn_32_32/ $DEFAULT_ARGS --convar_type naive

exit
}
