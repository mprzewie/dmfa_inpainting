# 2021-03-15 00:33:20.668684
python train_wae.py --experiment_name=new_metrics/wae_8_8/inp_acflow_appnd_mask --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --batch_size 128 --convar_append_mask
