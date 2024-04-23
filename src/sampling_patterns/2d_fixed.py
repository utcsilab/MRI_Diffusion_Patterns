import torch
import numpy as np

class Fixed2dPattern:
    def __init__(self, num_acs_lines, orientation, R, length, device):
        """
        A fixed 2d sampling pattern.
        
        Args:
            num_acs_lines (int): Number of ACS lines to keep in the center.
            orientation (str): Orientation of the sampling pattern. Either 'horizontal' or 'vertical'.
            R (int): Acceleration factor.
            length (int): Length of the 2d sampling pattern.
            device (torch.device): Device to store the mask on.
        """
        self.num_acs_lines = num_acs_lines
        self.orientation = orientation
        self.R = R
        self.length = length
        self.device = device
        
        # Initialize the mask
        center_line_idx = np.arange((length - num_acs_lines) // 2, (length + num_acs_lines) // 2)
        outer_line_idx = np.setdiff1d(np.arange(length), center_line_idx)
        
        outer_R = np.round((length - num_acs_lines) / (length/R - num_acs_lines))
        
        random_line_idx = outer_line_idx[::int(outer_R)]
        
        c = torch.zeros(length)
        c[center_line_idx] = 1.
        c[random_line_idx] = 1.
        
        self.weights = c.to(device=device, dtype=torch.float32)
    
    def sample_mask(self, n=1):
        """
        Returns the sampling mask.
        
        Args:
            n (int): Number of masks to sample.
        
        Returns:
            torch.Tensor: Sampling mask. [n, 1, length, length].
        """
        if self.orientation == "horizontal":
            raw_mask = self.weights.unsqueeze(1).repeat(1, self.length)
        elif self.orientation == "vertical":
            raw_mask = self.weights.unsqueeze(0).repeat(self.length, 1)
        
        return raw_mask.unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)
    