# 2021-05-02 00:28:11.688457
python train_wae.py --experiment_name=ae_tryout:rerun-2/inp_noise --convar_type=naive --inpainter_type=noise --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 80 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=2e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1 --wae_disc_loss_weight 0 --skip_fid
