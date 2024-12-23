python3 src/train.py gpu=0 exp_name=PDFS_DOGG_R4_K1_BS8 data=PDFS_whitened_knees data.train_batch_size=8 pattern=3D_DOGG pattern.R=4 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=1

python3 src/train.py gpu=0 exp_name=PDFS_DOGG_R8_K1_BS16 data=PDFS_whitened_knees data.train_batch_size=16 pattern=3D_DOGG pattern.R=8 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=1

python3 src/train.py gpu=0 exp_name=PDFS_DOGG_R12_K1_BS24 data=PDFS_whitened_knees data.train_batch_size=24 pattern=3D_DOGG pattern.R=12 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=1

python3 src/train.py gpu=0 exp_name=PDFS_DOGG_R16_K1_BS32 data=PDFS_whitened_knees data.train_batch_size=32 pattern=3D_DOGG pattern.R=16 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=1

python3 src/train.py gpu=0 exp_name=PDFS_DOGG_R20_K1_BS40 data=PDFS_whitened_knees data.train_batch_size=40 pattern=3D_DOGG pattern.R=20 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=1
