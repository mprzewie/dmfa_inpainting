# 2021-05-07 08:56:55.029885
python train_classifier.py --experiment_name=exp_new_classifier_v4/ce_comp --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/complete_data/context_encoder_v1/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --mask_train_size=14 --mask_val_size=14 --convar_type naive
