# 2021-04-30 01:19:49.765747
python train_classifier.py --experiment_name=exp_val_mask_7/inp_acflow --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --mask_train_size=14 --mask_val_size=7 --convar_type naive
