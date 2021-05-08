# 2021-04-18 00:29:13.143993
python train_classifier.py --experiment_name=exp_v1/inp_gt --convar_type=naive --inpainter_type=gt --dataset cifar10 --img_size 32 --mask_hidden_h 16 --mask_hidden_w 16 --num_epochs=10 --lr=1e-4 --dump_sample_results --render_every 3 --max_benchmark_batches=-1 --cls_bl 3 --cls_latent_size 256 --cls_dropout=0.5 --cls_depth 3
