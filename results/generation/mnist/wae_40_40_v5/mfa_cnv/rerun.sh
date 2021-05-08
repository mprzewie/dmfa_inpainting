# 2021-04-15 17:03:00.802169
python train_wae.py --experiment_name=wae_40_40_v5/mfa_cnv --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 40 --wae_lc 40 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1 --convar_type naive
