# 2021-04-15 00:40:34.233357
python train_wae.py --experiment_name=wae_40_40_v5/inp_gt --convar_type=naive --inpainter_type=gt --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 40 --wae_lc 40 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1
