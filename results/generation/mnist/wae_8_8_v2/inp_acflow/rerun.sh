# 2021-03-22 03:29:22.198438
python train_wae.py --experiment_name=wae_8_8_v2/inp_acflow --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --convar_type naive
