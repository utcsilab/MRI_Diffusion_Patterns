import torch
import numpy as np
from src.sampling_patterns.pattern_utils import (get_xy_radius_grid, 
                                                 get_random_logits, 
                                                 normalize_probs, 
                                                 bernouli_gumbel_sample, 
                                                 bernouli_straight_through_sample,
                                                 shape_mask_3d,
                                                 get_furthest_point_3d)

class Learned3d:
    def __init__(self, num_acs_lines, R, length, device, cut_corners, init_dist, sampler, tau=1.0):
        """
        A Learned 3D pattern.
        
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
        
        self.always_on_idx = flat_n_inds[acs_idx[:, None], acs_idx].flatten() #fancy indexing grabs a square from center
        
        #(2) Create the list of indexes to always keep off and on
        self.always_off_idx = np.empty(0, dtype=np.int64)
        if cut_corners:
            _, _, radius_grid = get_xy_radius_grid(length)
            self.always_off_idx = flat_n_inds[radius_grid > 1].flatten()
            
        self.insert_mask_idx = np.array([i for i in range(self.length**2) 
                                        if (i not in self.always_on_idx) and (i not in self.always_off_idx)])
                
        #(3) Initialize the weights
        self.weights = get_random_logits(length=self.num_weights, 
                                         dist=init_dist, 
                                         normalize=True, 
                                         norm_mean=self.sparsity_level).to(device=device, dtype=torch.float32)
        self.weights.requires_grad_()
    
    def parameters(self):
        """
        Returns the parameters of the pattern.
        
        Returns:
            List[Dict]: List of dictionaries containing the parameters.
        """
        return [{'params': self.weights}]

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
                                     on_idx=self.always_on_idx, 
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
                                  on_idx=self.always_on_idx, 
                                  off_idx=self.always_off_idx) #[1, length, length]
        
        return prob_mask.unsqueeze(0) #[1, 1, length, length]
    
    # @property
    # def insert_mask_idx(self):
    #     """
    #     Returns the list of flattened 2d indexes to insert the flattened pattern weights.
        
    #     Returns:
    #         np.ndarray: The indexes. [num_weights].
    #     """
    #     #Set up caching for faster future access
    #     if (not hasattr(self, 'cache_insert_mask_idx')) or (self.on_len != len(self.always_on_idx)) or (self.off_len != len(self.always_off_idx)):
    #         self.on_len = len(self.always_on_idx)
    #         self.off_len = len(self.always_off_idx)
    #         self.cache_insert_mask_idx = np.array([i for i in range(self.length**2) 
    #                                                if (i not in self.always_on_idx) and (i not in self.always_off_idx)])
        
    #     return self.cache_insert_mask_idx
    
    @property
    def sparsity_level(self):
        """
        Calculated and returned the current desired sparsity level of the learned weights.
        
        Returns:
            float: The current desired sparsity level.
        """
        total_on = self.length**2 / self.R
        already_on = len(self.always_on_idx)
        
        return (total_on - already_on) / self.num_weights
    
    @property
    def num_weights(self):
        """
        Returns the number of weights in the pattern.
        
        Returns:
            int: The number of weights.
        """
        return len(self.insert_mask_idx)
    
    def greedy_topk_step(self, k, include_conjugates=False):
        """
        Takes one step of greedy max-min neighbor optimization with a given k.
        Returns True if the desired acceleration is met, else returns False.

        Args:
            k (int, optional): Top-k parameter for greedy optimization. Defaults to 1.
            include_conjugates (bool, optional): Whether to include the conjugates of the existing points 
                                                     when considering nearest points. Defaults to False.
        
        Returns:
            finished_flag (bool): Flag that indicates whether the desired acceleration level is met.
        """
        if self.sparsity_level <= 0:
            return True
        
        if not hasattr(self, 'grid_x'):
            self.grid_x, self.grid_y, _ = get_xy_radius_grid(self.length)
            self.grid_x = self.grid_x.to(device=self.device)
            self.grid_y = self.grid_y.to(device=self.device)
        
        #Calculate the point that's (a) in the top-k negative gradients
        # and (b) furthest from the already selected points
        top_k_inds = torch.topk(-self.weights.grad, k=k)[1].tolist() 
        selected_point_idx = get_furthest_point_3d(query_point_inds=self.insert_mask_idx[top_k_inds],
                                                   key_point_inds=self.always_on_idx,
                                                   grid_x=self.grid_x,
                                                   grid_y=self.grid_y,
                                                   include_conjugate_keys=include_conjugates)
        top_k_inds = [top_k_inds[selected_point_idx]] 
        
        self.always_on_idx = np.append(self.always_on_idx, self.insert_mask_idx[top_k_inds])
        
        mask = np.ones(len(self.insert_mask_idx), dtype=bool)
        mask[top_k_inds] = False
        self.insert_mask_idx = self.insert_mask_idx[mask]
        
        #Remove the selected point from the weights and gradients
        keep_inds = [i for i in range(self.weights.numel()) if i not in top_k_inds]
        weights = torch.empty(len(keep_inds),
                              dtype=self.weights.dtype,
                              layout=self.weights.layout,
                              device=self.weights.device)
        weights.data = self.weights[keep_inds].data
        self.weights = weights
        self.weights.requires_grad_() #Trying to manually set here to ensure it's zerod out
        
        finished_flag = True if self.sparsity_level <= 0 else False
        return finished_flag
    