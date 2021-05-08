# 2021-03-01 09:54:19.634137
python train_wae.py --experiment_name=wae_n_blocks/inp_gt_1_bl --convar_type=naive --inpainter_type=gt --batch_size 64 --dataset svhn --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=10 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --wae_bl 1 --wae_recon_loss mse
