# 2021-04-16 15:16:11.613607
python train_wae.py --experiment_name=ae_tryout/inp_knn_5k --convar_type=naive --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 80 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=2e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1 --wae_disc_loss_weight 0 --skip_fid --inpainter_type=knn --convar_type naive
