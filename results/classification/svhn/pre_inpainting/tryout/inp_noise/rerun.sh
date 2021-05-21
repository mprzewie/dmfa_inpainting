# 2021-05-16 17:31:46.334921
python train_classifier_v2.py --experiment_name=tryout/inp_noise --convar_type=naive --inpainter_type=noise --dataset svhn --img_size 32 --mask_train_size 0 --mask_unknown_size 16 --mask_val_size 16 --num_epochs=25 --lr=1e-3 --dump_sample_results --render_every 3 --dataset_root /mnt/remote/wmii_gmum_projects/datasets/vision/SVHN/ --max_benchmark_batches=-1
