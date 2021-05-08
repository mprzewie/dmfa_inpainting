# 2021-02-28 10:58:30.753494
python train_wae.py --experiment_name=wae_2nd_experiments/inp_mfa_pretrained_inpainter_frozen --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/svhn_32_32/ --batch_size 64 --dataset svhn --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=10 --render_every 2 --dataset_root /mnt/users/mprzewiezlikowski/local/data/.data/
