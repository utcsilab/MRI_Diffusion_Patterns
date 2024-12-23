python3 src/train.py gpu=0 exp_name=COMBINED_LOUPE_R4 data=COMBINED_whitened_knees data.train_batch_size=1 pattern=3D_Loupe pattern.R=4 recon=DPS_COMBINED_knees_whitened training=3D_Loupe

python3 src/train.py gpu=0 exp_name=COMBINED_LOUPE_R8 data=COMBINED_whitened_knees data.train_batch_size=1 pattern=3D_Loupe pattern.R=8 recon=DPS_COMBINED_knees_whitened training=3D_Loupe

python3 src/train.py gpu=0 exp_name=COMBINED_LOUPE_R12 data=COMBINED_whitened_knees data.train_batch_size=1 pattern=3D_Loupe pattern.R=12 recon=DPS_COMBINED_knees_whitened training=3D_Loupe

python3 src/train.py gpu=0 exp_name=COMBINED_LOUPE_R16 data=COMBINED_whitened_knees data.train_batch_size=1 pattern=3D_Loupe pattern.R=16 recon=DPS_COMBINED_knees_whitened training=3D_Loupe

python3 src/train.py gpu=0 exp_name=COMBINED_LOUPE_R20 data=COMBINED_whitened_knees data.train_batch_size=1 pattern=3D_Loupe pattern.R=20 recon=DPS_COMBINED_knees_whitened training=3D_Loupe
