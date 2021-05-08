# 2021-03-21 16:33:25.733735
python train_classifier.py --experiment_name=exp_more_visualizations_v3/mfa_cnv --inpainter_type=mfa --inpainter_path ../../gmm_missing/models/mnist_28_28/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --convar_type naive
