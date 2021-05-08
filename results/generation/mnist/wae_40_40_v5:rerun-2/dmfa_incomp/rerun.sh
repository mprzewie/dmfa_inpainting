# 2021-05-02 22:30:16.072034
python train_wae.py --experiment_name=wae_40_40_v5:rerun-2/dmfa_incomp --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 40 --wae_lc 40 --dataset_root /home/mprzewiezlikowski/uj/.data --max_benchmark_batches=-1 --lr=4e-4 --batch_size=128 --wae_recon_loss mse --wae_bl 1
