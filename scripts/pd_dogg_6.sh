python3 src/train.py gpu=0 exp_name=PD_DOGG_R4_K16_BS4 data=PD_whitened_knees data.train_batch_size=4 pattern=3D_DOGG pattern.R=4 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=16

python3 src/train.py gpu=0 exp_name=PD_DOGG_R8_K32_BS8 data=PD_whitened_knees data.train_batch_size=8 pattern=3D_DOGG pattern.R=8 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=32

python3 src/train.py gpu=0 exp_name=PD_DOGG_R12_K48_BS12 data=PD_whitened_knees data.train_batch_size=12 pattern=3D_DOGG pattern.R=12 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=48

python3 src/train.py gpu=0 exp_name=PD_DOGG_R16_K64_BS16 data=PD_whitened_knees data.train_batch_size=16 pattern=3D_DOGG pattern.R=16 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=64

python3 src/train.py gpu=0 exp_name=PD_DOGG_R20_K80_BS20 data=PD_whitened_knees data.train_batch_size=20 pattern=3D_DOGG pattern.R=20 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=80
