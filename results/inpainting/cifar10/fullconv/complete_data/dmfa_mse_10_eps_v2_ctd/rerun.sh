# 2021-04-12 12:57:20.015667
python train_inpainter.py --dataset cifar10 --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --experiment_name fullconv/complete_data/dmfa_mse_10_eps_v2
# 2021-04-12 12:57:43.400148
python train_inpainter.py --dataset cifar10 --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --experiment_name fullconv/complete_data/dmfa_mse_10_eps_v2
# 2021-04-12 13:45:13.341994
python train_inpainter.py --dataset cifar10 --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0 --experiment_name fullconv/complete_data/dmfa_mse_10_eps_v2
# 2021-05-11 00:38:47.835415
python train_inpainter.py --dataset cifar10 --img_size 32 --mask_train_size 16 --mask_val_size 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 50 --l_nll_weight 1 --l_mse_weight 0 --experiment_name fullconv/complete_data/dmfa_mse_10_eps_v2_ctd
