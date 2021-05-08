#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=1

EXPERIMENTS_DIR=ae_tryout:rerun-2

DEFAULT_ARGS="--dataset mnist  --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 80 --dataset_root /home/mprzewiezlikowski/uj/.data \
    --max_benchmark_batches=-1 --lr=2e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1 --wae_disc_loss_weight 0 --skip_fid" 

# EXPERIMENTS_DIR=wae_40_40_v5:rerun-2

# DEFAULT_ARGS="--dataset mnist --num_epochs=10 --render_every 2 --wae_fc 40 --wae_lc 40 --dataset_root /home/mprzewiezlikowski/uj/.data \
#     --max_benchmark_batches=-1 --lr=4e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1"


# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_gt --convar_type=naive --inpainter_type=gt  $DEFAULT_ARGS 
    
# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_zero --convar_type=naive --inpainter_type=zero ${DEFAULT_ARGS}

# python train_wae.py \
#     --experiment_name=$EXPERIMENTS_DIR/inp_noise --convar_type=naive --inpainter_type=noise ${DEFAULT_ARGS}

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_comp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/complete_data/v1/ ${DEFAULT_ARGS} 

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_comp_cnv \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/complete_data/v1/ ${DEFAULT_ARGS} \
    --convar_type naive

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ ${DEFAULT_ARGS} 

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp_cnv \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ ${DEFAULT_ARGS} \
    --convar_type naive

conda activate acflow

python train_wae.py --experiment_name=$EXPERIMENTS_DIR/inp_acflow \
    --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ ${DEFAULT_ARGS} \
    --convar_type naive

conda activate uj

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ ${DEFAULT_ARGS} 

python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa_cnv \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ ${DEFAULT_ARGS} \
    --convar_type naive



python train_wae.py \
        --experiment_name=$EXPERIMENTS_DIR/inp_knn_5k --convar_type=naive ${DEFAULT_ARGS} \
        --inpainter_type=knn --convar_type naive


python train_wae.py \
    --experiment_name=$EXPERIMENTS_DIR/partial_conv --convar_type=partial --inpainter_type=zero ${DEFAULT_ARGS}

