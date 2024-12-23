python3 src/train.py gpu=2 exp_name=PD_DOGG_R4_K32_BS8 data=PD_whitened_knees data.train_batch_size=8 pattern=3D_DOGG pattern.R=4 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=32

python3 src/train.py gpu=2 exp_name=PD_DOGG_R8_K64_BS16 data=PD_whitened_knees data.train_batch_size=16 pattern=3D_DOGG pattern.R=8 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=64

python3 src/train.py gpu=2 exp_name=PD_DOGG_R12_K96_BS24 data=PD_whitened_knees data.train_batch_size=24 pattern=3D_DOGG pattern.R=12 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=96

python3 src/train.py gpu=2 exp_name=PD_DOGG_R16_K128_BS32 data=PD_whitened_knees data.train_batch_size=32 pattern=3D_DOGG pattern.R=16 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=128

python3 src/train.py gpu=2 exp_name=PD_DOGG_R20_K160_BS40 data=PD_whitened_knees data.train_batch_size=40 pattern=3D_DOGG pattern.R=20 recon=DPS_PD_knees_whitened training=3D_DOGG training.k=160
