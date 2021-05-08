# 2021-03-29 00:39:42.669900
python train_wae.py --experiment_name=wae_8_8_fake_sampling/dmfa_comp --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/complete_data/v1/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --batch_size 128 --num_epochs=2 --lr 2e-4
