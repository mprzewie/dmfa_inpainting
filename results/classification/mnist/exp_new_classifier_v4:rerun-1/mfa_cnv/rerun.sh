# 2021-05-02 00:35:57.511924
python train_classifier.py --experiment_name=exp_new_classifier_v4:rerun-1/mfa_cnv --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --mask_train_size=14 --mask_val_size=14 --convar_type naive
