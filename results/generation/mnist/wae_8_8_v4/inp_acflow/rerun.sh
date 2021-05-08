# 2021-04-08 19:20:37.913481
python train_wae.py --experiment_name=wae_8_8_v4/inp_acflow --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-3 --batch_size=128 --convar_type naive
