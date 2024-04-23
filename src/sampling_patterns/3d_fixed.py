import torch
import numpy as np

class Fixed3dPattern:
    def __init__(self, num_acs_lines, R, length, device):
        """
        A fixed 3d sampling pattern.
        
        Args:
            num_acs_lines (int): Number of ACS lines to keep in the center.
            R (int): Acceleration factor.
            length (int): Length of the 1D sampling pattern.
            device (torch.device): Device to store the mask on.
        """
        