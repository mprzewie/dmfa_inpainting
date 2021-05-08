# 2021-04-15 01:28:56.223071
python train_wae.py --experiment_name=wae_40_40_v5/inp_noise --convar_type=naive --inpainter_type=noise --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 40 --wae_lc 40 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1
