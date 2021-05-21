# 2021-05-16 23:03:37.582065
python train_classifier_v2.py --experiment_name=tryout/dmfa_incomp_cnv --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/incomplete_data/v1/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --mask_train_size=14 --mask_val_size=14 --convar_type naive
