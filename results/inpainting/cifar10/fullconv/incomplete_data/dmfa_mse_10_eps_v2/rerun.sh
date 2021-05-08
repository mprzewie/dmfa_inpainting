# 2021-04-12 18:39:01.717367
python train_inpainter.py --dataset cifar10 --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1 --mask_unknown_size 16 --experiment_name fullconv/incomplete_data/dmfa_mse_10_eps_v2
# 2021-04-12 19:10:40.296744
python train_inpainter.py --dataset cifar10 --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0 --mask_unknown_size 16 --experiment_name fullconv/incomplete_data/dmfa_mse_10_eps_v2
