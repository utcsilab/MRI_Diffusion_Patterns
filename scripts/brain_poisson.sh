python3 src/eval.py gpu=0 exp_name=brain_Poisson_R4 data=T2_whitened_brains pattern=3D_Poisson pattern.R=4 recon=DPS_brains_whitened

python3 src/eval.py gpu=0 exp_name=brain_Poisson_R8 data=T2_whitened_brains pattern=3D_Poisson pattern.R=8 recon=DPS_brains_whitened

python3 src/eval.py gpu=0 exp_name=brain_Poisson_R12 data=T2_whitened_brains pattern=3D_Poisson pattern.R=12 recon=DPS_brains_whitened

python3 src/eval.py gpu=0 exp_name=brain_Poisson_R16 data=T2_whitened_brains pattern=3D_Poisson pattern.R=16 recon=DPS_brains_whitened

python3 src/eval.py gpu=0 exp_name=brain_Poisson_R20 data=T2_whitened_brains pattern=3D_Poisson pattern.R=20 recon=DPS_brains_whitened
