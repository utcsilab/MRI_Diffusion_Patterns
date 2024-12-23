python3 src/train.py gpu=2 exp_name=PDFS_DOGG_R4_K2_BS8 data=PDFS_whitened_knees data.train_batch_size=8 pattern=3D_DOGG pattern.R=4 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=2

python3 src/train.py gpu=2 exp_name=PDFS_DOGG_R8_K4_BS16 data=PDFS_whitened_knees data.train_batch_size=16 pattern=3D_DOGG pattern.R=8 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=4

python3 src/train.py gpu=2 exp_name=PDFS_DOGG_R12_K6_BS24 data=PDFS_whitened_knees data.train_batch_size=24 pattern=3D_DOGG pattern.R=12 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=6

python3 src/train.py gpu=2 exp_name=PDFS_DOGG_R16_K8_BS32 data=PDFS_whitened_knees data.train_batch_size=32 pattern=3D_DOGG pattern.R=16 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=8

python3 src/train.py gpu=2 exp_name=PDFS_DOGG_R20_K10_BS40 data=PDFS_whitened_knees data.train_batch_size=40 pattern=3D_DOGG pattern.R=20 recon=DPS_PDFS_knees_whitened training=3D_DOGG training.k=10
