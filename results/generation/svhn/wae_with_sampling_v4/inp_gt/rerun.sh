# 2021-04-06 19:52:02.383044
python train_wae.py --experiment_name=wae_with_sampling_v4/inp_gt --convar_type=naive --inpainter_type=gt --batch_size 64 --dataset svhn --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=10 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --lr 4e-4 --wae_bl 2 --wae_recon_loss mse --max_benchmark_batches=-1
