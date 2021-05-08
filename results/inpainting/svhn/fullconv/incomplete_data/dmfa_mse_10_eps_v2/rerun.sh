# 2021-04-08 09:29:04.749144
python train_inpainter.py --experiment_name fullconv/incomplete_data/dmfa_mse_10_eps_v2 --dataset svhn --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --mask_unknown_size 16 --num_epochs 10 --l_nll_weight 1 --l_mse_weight 1
# 2021-04-08 10:21:10.119480
python train_inpainter.py --experiment_name fullconv/incomplete_data/dmfa_mse_10_eps_v2 --dataset svhn --img_size 32 --mask_hidden_w 16 --mask_hidden_h 16 --batch_size 64 --architecture fullconv --bkb_fc 32 --bkb_lc 32 --bkb_depth 3 --bkb_block_length 2 --num_factors 4 --a_amplitude 0.2 --mask_unknown_size 16 --num_epochs 90 --l_nll_weight 1 --l_mse_weight 0
