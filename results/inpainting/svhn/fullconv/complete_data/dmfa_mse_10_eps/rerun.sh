# 2021-03-29 19:15:52.853179
python train_inpainter.py --experiment_name fullconv/complete_data/dmfa_mse_10_eps --dataset svhn --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1
# 2021-03-29 20:29:42.967868
python train_inpainter.py --experiment_name fullconv/complete_data/dmfa_mse_10_eps --dataset svhn --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0
