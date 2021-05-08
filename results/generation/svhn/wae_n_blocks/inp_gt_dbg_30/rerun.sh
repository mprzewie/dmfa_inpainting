# 2021-03-01 18:57:17.804762
python train_wae.py --experiment_name=wae_n_blocks/inp_gt_dbg --convar_type=naive --inpainter_type=gt --batch_size 64 --dataset svhn --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=30 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/ --wae_bl 2 --wae_recon_loss mse
