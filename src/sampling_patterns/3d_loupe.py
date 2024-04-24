import torch
import numpy as np
from src.sampling_patterns.pattern_utils import (get_xy_radius_grid, 
                                                 get_random_logits, 
                                                 normalize_probs, 
                                                 bernouli_gumbel_sample, 
                                                 bernouli_straight_through_sample,
                                                 shape_mask_3d)

class Loupe3d:
    def __init__(self, num_acs_lines, R, length, device, cut_corners, init_dist, sampler, tau=1.0):
        """
        A LOUPE-parameterized 3D pattern.
        
        Args:
            num_acs_lines (int): Number of ACS lines to keep in the center.
            R (int): Acceleration factor.
            length (int): Length of the 3D sampling pattern.
            device (torch.device): Device to store the mask on.
            cut_corners (bool): Whether to cut the corners of the 3D sampling pattern.
            init_dist (str): The distribution to sample the initial logits from. Options in [normal, uniform, random].
            sampler (str): The sampling method to use. Options in [gumbel, straight_through].
            tau (float): The temperature parameter for the Gumbel-Softmax distribution. Not used unless sampler is gumbel. 
                         Default is 1.0.
        """
        self.num_acs_lines = num_acs_lines
        self.R = R
        self.length = length
        self.device = device
        self.cut_corners = cut_corners
        self.init_dist = init_dist
        self.sampler = sampler
        self.tau = tau
        
        # Initialize the mask
        #(1) Set the location and number of the ACS lines
        acs_idx = np.arange((length - num_acs_lines) // 2, (length + num_acs_lines) // 2)
        flat_n_inds = np.arange(length**2).reshape(length, length)
        
        self.acs_idx = flat_n_inds[acs_idx[:, None], acs_idx].flatten() #fancy indexing grabs a square from center
        
        #(2) Create the list of indexes to always keep off
        self.always_off_idx = np.empty(0, dtype=np.int64)
        if cut_corners:
            _, _, radius_grid = get_xy_radius_grid(length)
            self.always_off_idx = flat_n_inds[radius_grid > 1].flatten()
        
        #(3) Create the list of flattened 2d indexes to insert the flattened pattern weights
        self.insert_mask_idx = np.array([i for i in range(length**2) if i not in self.acs_idx and i not in self.always_off_idx])
        
        self.num_weights = len(self.insert_mask_idx)
        
        self.sparsity_level = ((length**2)/R - num_acs_lines**2) / self.num_weights #proportion of ones desired from the weights; informs the mean probability
        
        #(4) Initialize the weights
        self.weights = get_random_logits(length=self.num_weights, 
                                         dist=init_dist, 
                                         normalize=True, 
                                         norm_mean=self.sparsity_level).to(device=device, dtype=torch.float32)
        self.weights.requires_grad_()
    
    def sample_mask(self, n=1):
        """
        Samples a pattern using the learned weights.
        
        Args:
            n (int): Number of masks to sample.
            
        Returns:
            torch.Tensor: Sampling mask. [n, 1, length, length]. 
        """
        probs = torch.sigmoid(self.weights)
        normed_probs = normalize_probs(probs=probs, mean=self.sparsity_level)
        normed_probs = normed_probs.unsqueeze(0).repeat(n, 1) #[n, num_weights]
        if self.sampler == 'gumbel':
            flat_sample = bernouli_gumbel_sample(probs=normed_probs, tau=self.tau)
        elif self.sampler == 'straight_through':
            flat_sample = bernouli_straight_through_sample(probs=normed_probs)
        sampled_mask = shape_mask_3d(length=self.length, 
                                     flat_input=flat_sample, 
                                     flat_input_idx=self.insert_mask_idx, 
                                     on_idx=self.acs_idx, 
                                     off_idx=self.always_off_idx) #[n, length, length]
        
        return sampled_mask.unsqueeze(1) #[n, 1, length, length]
    
    @torch.no_grad()
    def probabilistic_mask(self):
        """
        Returns the probabilistic mask.
        
        Returns:
            torch.Tensor: The probabilistic mask. [1, 1, length, length].
        """
        probs = torch.sigmoid(self.weights)
        normed_probs = normalize_probs(probs=probs, mean=self.sparsity_level)
        prob_mask = shape_mask_3d(length=self.length, 
                                  flat_input=normed_probs, 
                                  flat_input_idx=self.insert_mask_idx, 
                                  on_idx=self.acs_idx, 
                                  off_idx=self.always_off_idx) #[1, length, length]
        
        return prob_mask.unsqueeze(0) #[1, 1, length, length]
    