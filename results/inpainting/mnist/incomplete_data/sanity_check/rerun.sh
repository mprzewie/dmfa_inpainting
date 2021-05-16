# 2021-05-10 13:15:53.383214
python train_inpainter.py --experiment_name incomplete_data/sanity_check --batch_size=48 --mask_unknown_size=14 --num_epochs=5 --render_every 1
# 2021-05-10 13:17:19.520981
python train_inpainter.py --experiment_name incomplete_data/sanity_check --batch_size=128 --mask_unknown_size=14 --num_epochs=5 --render_every 1
# 2021-05-10 13:22:54.535291
python train_inpainter.py --experiment_name incomplete_data/sanity_check --batch_size=128 --mask_unknown_size=14 --num_epochs=5 --render_every 1 --max_benchmark_batches 5
