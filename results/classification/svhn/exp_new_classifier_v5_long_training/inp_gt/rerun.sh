# 2021-04-30 11:23:55.798036
python train_classifier.py --experiment_name=exp_new_classifier_v5_long_training/inp_gt --convar_type=naive --inpainter_type=gt --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1
