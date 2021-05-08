# 2021-04-30 14:18:27.189680
python train_classifier.py --experiment_name=exp_new_classifier_v5_long_training/inp_noise --convar_type=naive --inpainter_type=noise --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1
