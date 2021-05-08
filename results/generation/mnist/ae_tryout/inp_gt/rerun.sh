# 2021-04-15 01:31:54.159714
python train_wae.py --experiment_name=ae_tryout/inp_gt --convar_type=naive --inpainter_type=gt --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 80 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=2e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1 --wae_disc_loss_weight 0 --skip_fid
