# 2021-05-16 23:54:54.862797
python train_wae_v2.py --experiment_name=wae_40_40/inp_gt --convar_type=naive --inpainter_type=gt --dataset mnist --num_epochs=20 --render_every 2 --wae_fc 40 --wae_lc 40 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1 --mask_train_size 0 --mask_unknown_size 14 --mask_val_size 14
