# 2021-03-13 03:19:01.740143
python train_classifier.py --experiment_name=exp_more_visualizations/apnd_mask/inp_acflow --inpainter_type=acflow --inpainter_path ../../ACFlow/exp/mnist/rnvp/ --dataset mnist --num_epochs=10 --lr=1e-3 --dump_sample_results --render_every 3 --convar_append_mask --batch_size 16
