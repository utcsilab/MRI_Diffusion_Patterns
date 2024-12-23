python3 src/train.py gpu=1 exp_name=PD_DOGG_R4_K16_BS8 data=PD_whitened_knees data.train_batch_size=8 pattern=3D_DOGG pattern.R=4 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=16

python3 src/train.py gpu=1 exp_name=PD_DOGG_R8_K32_BS16 data=PD_whitened_knees data.train_batch_size=16 pattern=3D_DOGG pattern.R=8 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=32

python3 src/train.py gpu=1 exp_name=PD_DOGG_R12_K48_BS24 data=PD_whitened_knees data.train_batch_size=24 pattern=3D_DOGG pattern.R=12 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=48

python3 src/train.py gpu=1 exp_name=PD_DOGG_R16_K64_BS32 data=PD_whitened_knees data.train_batch_size=32 pattern=3D_DOGG pattern.R=16 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=64

python3 src/train.py gpu=1 exp_name=PD_DOGG_R20_K80_BS40 data=PD_whitened_knees data.train_batch_size=40 pattern=3D_DOGG pattern.R=20 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=80
