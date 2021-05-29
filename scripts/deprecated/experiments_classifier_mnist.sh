#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=3

EXPERIMENTS_DIR=exp_new_classifier_v4:rerun-1

DEFAULT_ARGS="--dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 \
            --mask_train_size=14 --mask_val_size=14"


python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_gt --convar_type=naive --inpainter_type=gt $DEFAULT_ARGS 
    
python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_zero --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS 

python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/inp_noise --convar_type=naive --inpainter_type=noise $DEFAULT_ARGS 

python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_comp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/complete_data/v1/ $DEFAULT_ARGS 

python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_comp_cnv \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/complete_data/v1/ $DEFAULT_ARGS \
    --convar_type naive

python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ $DEFAULT_ARGS 

python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/dmfa_incomp_cnv \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ $DEFAULT_ARGS \
    --convar_type naive



set +x
conda activate acflow

set -x

python train_classifier.py --experiment_name=$EXPERIMENTS_DIR/inp_acflow \
    --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ $DEFAULT_ARGS \
    --convar_type naive


conda activate uj
    
python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/partial_conv --convar_type=partial --inpainter_type=zero $DEFAULT_ARGS 


python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ $DEFAULT_ARGS 

python train_classifier.py \
    --experiment_name=$EXPERIMENTS_DIR/mfa_cnv \
    --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ $DEFAULT_ARGS --convar_type naive


python train_classifier.py \
        --experiment_name=$EXPERIMENTS_DIR/inp_knn_5k --convar_type=naive $DEFAULT_ARGS \
        --inpainter_type=knn --convar_type naive