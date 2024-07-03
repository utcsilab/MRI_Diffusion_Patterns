import torch
import numpy as np
import sigpy.mri

class Fixed3dPattern:
    def __init__(self, num_acs_lines, R, image_size, device, cut_corners, seed, pad_size=None):
        """
        A fixed 3d sampling pattern.
        
        Args:
            num_acs_lines (int): Number of ACS lines to keep in the center.
            R (int): Acceleration factor.
            image_size (tuple of ints): Size of the k-space to be used with this pattern. [H, W].
            device (torch.device): Device to store the mask on.
            cut_corners (bool): Whether to cut the corners of the 3D sampling pattern.
            seed (int): Seed for the random number generator.
            pad_size (tuple of ints): Size of the larger final pattern if padding is desired.
                                      If given, both dimensions must be larger than or equal to the image size.
                                      Optional, defaults to None in which case no padding is done.
        """
        self.num_acs_lines = num_acs_lines
        self.R = R
        self.image_size = image_size
        self.device = device
        self.cut_corners = cut_corners
        self.seed = seed
        self.pad_size = pad_size
        
        H, W = image_size
        
        # Initialize the mask
        if H == W:
            c = sigpy.mri.poisson(img_shape=(H, W),
                                accel=R,
                                calib=(num_acs_lines, num_acs_lines),
                                dtype=np.float32,
                                crop_corner=cut_corners,
                                seed=seed + 1 if seed==2023 else seed) #hangs if seed is 2023
        else:
            c = sigpy.mri.poisson(img_shape=(H, W),
                                accel=R,
                                dtype=np.float32,
                                crop_corner=cut_corners,
                                seed=seed + 1 if seed==2023 else seed) #hangs if seed is 2023
            
            start_row = (H - num_acs_lines) // 2
            start_col = (W - num_acs_lines) // 2
            end_row = start_row + num_acs_lines
            end_col = start_col + num_acs_lines
            
            c[start_row:end_row, start_col:end_col] = 1.0
            
        # Pad the mask if necessary
        if pad_size is not None:
            H_pad, W_pad = pad_size
            H_diff = H_pad - H
            W_diff = W_pad - W
            
            c = np.pad(c, ((H_diff // 2, H_diff - H_diff // 2), (W_diff // 2, W_diff - W_diff // 2)), mode='constant', constant_values=0)
        
        self.weights = torch.from_numpy(c).to(device=device, dtype=torch.float32)
    
    def sample_mask(self, n=1):
        """
        Returns the sampling mask.
        
        Args:
            n (int): Number of masks to sample.
        
        Returns:
            torch.Tensor: Sampling mask. [n, 1, H, W].
        """
        return self.weights.unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)
    