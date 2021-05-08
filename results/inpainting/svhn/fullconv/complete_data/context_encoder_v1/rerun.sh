# 2021-05-05 22:07:42.544468
python train_inpainter.py --experiment_name fullconv/complete_data/context_encoder_v1 --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 1 --a_amplitude 0.2 --num_epochs 100 --l_nll_weight 0 --l_mse_weight 1 --lr 4e-4
