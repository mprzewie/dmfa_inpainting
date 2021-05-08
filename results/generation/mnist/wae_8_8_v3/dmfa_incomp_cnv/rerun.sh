# 2021-04-04 06:10:41.545021
python train_wae.py --experiment_name=wae_8_8_v3/dmfa_incomp_cnv --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-3 --batch_size=128 --convar_type naive
