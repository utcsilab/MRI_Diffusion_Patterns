python3 src/train.py gpu=2 exp_name=COMBINED_DOGG_R4_K8_BS4 data=COMBINED_whitened_knees data.train_batch_size=4 pattern=3D_DOGG pattern.R=4 recon=DPS_COMBINED_knees_whitened training=3D_DOGG training.k=8

python3 src/train.py gpu=2 exp_name=COMBINED_DOGG_R8_K16_BS8 data=COMBINED_whitened_knees data.train_batch_size=8 pattern=3D_DOGG pattern.R=8 recon=DPS_COMBINED_knees_whitened training=3D_DOGG training.k=16

python3 src/train.py gpu=2 exp_name=COMBINED_DOGG_R12_K24_BS12 data=COMBINED_whitened_knees data.train_batch_size=12 pattern=3D_DOGG pattern.R=12 recon=DPS_COMBINED_knees_whitened training=3D_DOGG training.k=24

python3 src/train.py gpu=2 exp_name=COMBINED_DOGG_R16_K32_BS16 data=COMBINED_whitened_knees data.train_batch_size=16 pattern=3D_DOGG pattern.R=16 recon=DPS_COMBINED_knees_whitened training=3D_DOGG training.k=32

python3 src/train.py gpu=2 exp_name=COMBINED_DOGG_R20_K40_BS20 data=COMBINED_whitened_knees data.train_batch_size=20 pattern=3D_DOGG pattern.R=20 recon=DPS_COMBINED_knees_whitened training=3D_DOGG training.k=40
