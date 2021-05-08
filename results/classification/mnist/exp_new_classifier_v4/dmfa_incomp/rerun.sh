# 2021-04-30 13:30:11.380709
python train_classifier.py --experiment_name=exp_new_classifier_v4/dmfa_incomp --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --mask_train_size=14 --mask_val_size=14
