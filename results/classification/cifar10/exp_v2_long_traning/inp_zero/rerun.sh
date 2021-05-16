# 2021-05-10 18:09:22.488319
python train_classifier.py --experiment_name=exp_v2_long_traning/inp_zero --convar_type=naive --inpainter_type=zero --dataset cifar10 --img_size 32 --mask_train_size 16 --mask_val_size 16 --num_epochs=25 --lr=1e-4 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --cls_bl 3 --cls_latent_size 256 --cls_dropout=0.5 --cls_depth 3
