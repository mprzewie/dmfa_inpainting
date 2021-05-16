# 2021-05-10 23:33:30.871125
python train_classifier.py --experiment_name=exp_new_classifier_val_size_8/inp_knn_5k --convar_type=naive --inpainter_type=knn --dataset svhn --img_size 32 --mask_train_size 16 --mask_val_size 8 --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --dataset_root /home/mprzewie/uj/.data --batch_size 48
