#!/usr/bin/env bash


source ~/.bashrc
conda activate uj 

# set -e 
set -xe

export CUDA_VISIBLE_DEVICES=0

EXPERIMENTS_DIR=paper_experiment

DEFAULT_ARGS="--dataset svhn --img_size 32 --mask_train_size 0 --mask_unknown_size 16 --num_epochs=25 --lr=1e-3 --dump_sample_results \
                --render_every 25 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 "


{
  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/inp_gt --convar_type=naive --inpainter_type=gt $DEFAULT_ARGS

  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/inp_zero --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS

  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/inp_noise --convar_type=naive --inpainter_type=noise $DEFAULT_ARGS


  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/inp_dmfa_fc_incomp \
      --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/dmfa_mse_10_eps_v4_train_det/ $DEFAULT_ARGS

  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/inp_dmfa_fc_incomp_cnv \
      --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/dmfa_mse_10_eps_v4_train_det/ $DEFAULT_ARGS --convar_type naive

  conda activate acflow


  python train_classifier_v2.py --experiment_name=$EXPERIMENTS_DIR/inp_acflow \
      --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/svhn/rnvp/ $DEFAULT_ARGS

  conda activate uj


  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/partial_conv --convar_type=partial --inpainter_type=zero $DEFAULT_ARGS

  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/inp_zero_appnd --convar_type=naive --inpainter_type=zero $DEFAULT_ARGS --convar_append_mask



  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/mfa_frozen \
      --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/svhn_32_32/ $DEFAULT_ARGS

  python train_classifier_v2.py \
      --experiment_name=$EXPERIMENTS_DIR/mfa_frozen_cnv \
      --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/svhn_32_32/ $DEFAULT_ARGS \
      --convar_type naive

  conda activate rapids

  python train_classifier_v2.py \
          --experiment_name=$EXPERIMENTS_DIR/inp_knn_5k --convar_type=naive $DEFAULT_ARGS \
          --inpainter_type=knn

  exit
}