# 2021-03-21 17:50:44.149924
python train_wae.py --experiment_name=wae_2_bl_mse_bigger_lr_v2/mfa --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/svhn_32_32/ --batch_size 64 --dataset svhn --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=10 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --lr 4e-4 --wae_bl 2 --wae_recon_loss mse
