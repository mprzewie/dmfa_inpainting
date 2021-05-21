# 2021-05-16 15:52:08.645848
python train_classifier_v2.py --experiment_name=tryout/inp_gt --convar_type=naive --inpainter_type=gt --dataset cifar10 --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=50 --lr=1e-4 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --cls_bl 3 --cls_latent_size 256 --cls_dropout=0.5 --cls_depth 3
