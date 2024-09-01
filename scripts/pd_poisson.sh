python3 src/eval.py gpu=0 exp_name=PD_Poisson_R4 data=PD_whitened_knees pattern=3D_Poisson pattern.R=4.05 recon=DPS_PD_knees_whitened 

python3 src/eval.py gpu=0 exp_name=PD_Poisson_R8 data=PD_whitened_knees pattern=3D_Poisson pattern.R=8 recon=DPS_PD_knees_whitened

python3 src/eval.py gpu=0 exp_name=PD_Poisson_R12 data=PD_whitened_knees pattern=3D_Poisson pattern.R=12.1 recon=DPS_PD_knees_whitened 

python3 src/eval.py gpu=0 exp_name=PD_Poisson_R16 data=PD_whitened_knees pattern=3D_Poisson pattern.R=16.35 recon=DPS_PD_knees_whitened 

python3 src/eval.py gpu=0 exp_name=PD_Poisson_R20 data=PD_whitened_knees pattern=3D_Poisson pattern.R=20.6 recon=DPS_PD_knees_whitened
