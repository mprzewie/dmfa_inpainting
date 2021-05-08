#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 

export CUDA_VISIBLE_DEVICES=2

{

# python train_classifier.py \
#     --experiment_name=exp_new_classifier_v5_long_training/ce_comp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/context_encoder_v1/ \
#     --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 --convar_type naive

# python train_classifier.py \
#     --experiment_name=exp_new_classifier_v5_long_training/ce_incomp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/context_encoder_v1/ \
#     --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 --convar_type naive


# python train_classifier.py \
#     --experiment_name=exp_new_classifier_val_size_8/ce_comp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/context_encoder_v1/ \
#     --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 8 --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 --convar_type naive

# python train_classifier.py \
#     --experiment_name=exp_new_classifier_val_size_8/ce_incomp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/context_encoder_v1/ \
#     --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 8 --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 --convar_type naive


# python train_classifier.py \
#     --experiment_name=exp_new_classifier_val_size_0/ce_comp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/context_encoder_v1/ \
#     --dataset svhn --batch_size=64 --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=25 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 --convar_type naive

# python train_classifier.py \
#     --experiment_name=exp_new_classifier_val_size_0/ce_incomp \
#     --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/context_encoder_v1/ \
#     --dataset svhn --batch_size=64 --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=25 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1 --convar_type naive


python train_wae.py \
    --experiment_name=ae_v3_long_training/ce_comp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/complete_data/context_encoder_v1/ \
    --batch_size 64 --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --lr 2e-4 --wae_disc_loss_weight 0 --skip_fid --wae_recon_loss mse --wae_bl 2 --wae_depth 2 --max_benchmark_batches -1 --wae_latent_size 60 --wae_fc 32 --wae_lc 96 --convar_type naive


python train_wae.py \
    --experiment_name=ae_v3_long_training/ce_incomp \
    --inpainter_type=dmfa --inpainter_path ../results/inpainting/svhn/fullconv/incomplete_data/context_encoder_v1/ \
    --batch_size 64 --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --lr 2e-4 --wae_disc_loss_weight 0 --skip_fid --wae_recon_loss mse --wae_bl 2 --wae_depth 2 --max_benchmark_batches -1 --wae_latent_size 60 --wae_fc 32 --wae_lc 96 --convar_type naive


exit
}