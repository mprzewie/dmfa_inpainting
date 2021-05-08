# 2021-02-27 20:39:50.641668
python train_wae.py --experiment_name=new_metrics/wae_8_8/inp_dmfa_linear_heads_pretrained_inpainter_frozen --inpainter_type=dmfa --inpainter_path ../results/inpainting/mnist/mgr_sanity_check_v1/ --dataset mnist --num_epochs=10 --render_every 2 --wae_fc 8 --wae_lc 8
