# 2021-05-19 20:30:20.373539
python train_classifier_v2.py --experiment_name=tryout/partial --convar_type=partial --inpainter_type=zero --dataset cifar10 --img_size 32 --mask_train_size 0 --mask_unknown_size 16 --mask_val_size 16 --num_epochs=25 --lr=1e-4 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --cls_bl 3 --cls_latent_size 256 --cls_dropout=0.5 --cls_depth 3
