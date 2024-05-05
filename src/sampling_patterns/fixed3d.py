import torch
import numpy as np
import sigpy.mri

class Fixed3dPattern:
    def __init__(self, num_acs_lines, R, length, device, cut_corners, seed):
        """
        A fixed 3d sampling pattern.
        
        Args:
            num_acs_lines (int): Number of ACS lines to keep in the center.
            R (int): Acceleration factor.
            length (int): Length of the 3D sampling pattern.
            device (torch.device): Device to store the mask on.
            cut_corners (bool): Whether to cut the corners of the 3D sampling pattern.
            seed (int): Seed for the random number generator.
        """
        self.num_acs_lines = num_acs_lines
        self.R = R
        self.length = length
        self.device = device
        self.cut_corners = cut_corners
        self.seed = seed
        
        # Initialize the mask
        c = sigpy.mri.poisson(img_shape=(length, length),
                              accel=R,
                              calib=(num_acs_lines, num_acs_lines),
                              dtype=np.float32,
                              crop_corner=cut_corners,
                              seed=seed + 1 if seed==2023 else seed) #hangs if seed is 2023
        
        self.weights = torch.from_numpy(c).to(device=device, dtype=torch.float32)
    
    def sample_mask(self, n=1):
        """
        Returns the sampling mask.
        
        Args:
            n (int): Number of masks to sample.
        
        Returns:
            torch.Tensor: Sampling mask. [n, 1, length, length].
        """
        return self.weights.unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)
    