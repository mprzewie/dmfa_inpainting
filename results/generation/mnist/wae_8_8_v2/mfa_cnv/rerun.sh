# 2021-03-21 16:31:29.721679
python train_wae.py --experiment_name=wae_8_8_v2/mfa_cnv --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --convar_type naive
