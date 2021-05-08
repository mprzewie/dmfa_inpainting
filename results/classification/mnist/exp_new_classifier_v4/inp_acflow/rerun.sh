# 2021-04-30 14:01:38.054309
python train_classifier.py --experiment_name=exp_new_classifier_v4/inp_acflow --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --mask_train_size=14 --mask_val_size=14 --convar_type naive
