#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=3


 python train_inpainter.py --experiment_name complete_data/v1--batch_size=48


 python train_inpainter.py --experiment_name incomplete_data/v1 --batch_size=48 --mask_unknown_size=14

