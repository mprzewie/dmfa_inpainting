# 2021-04-04 07:26:38.848327
python train_wae.py --experiment_name=wae_8_8_v3/mfa --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-3 --batch_size=128
