import torch
import torch.nn.functional as F
import numpy as np

def get_xy_radius_grid(length):
    """
    Generate a grid of x and y coordinates for a square grid of radius length.
    Coordinates are centered at (0, 0) and normalized to the range [-1, 1].
    Also gives the radius of each point in the grid.
    
    Args:
        length (int): The radius of the grid.
    
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): The x and y coordinates of the grid, and the radius of each point.
    """
    x = y = (torch.arange(length) - (length-1)/2) / ((length-1)/2) #[length]
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy') #[length, length]
    grid = torch.stack([grid_x, grid_y], dim=0) #[2, length, length]
    radius_grid = torch.sqrt(torch.sum(torch.square(grid), dim=0)) #[length, length]
    
    return grid_x, grid_y, radius_grid

def normalize_probs(probs, mean):
    """
    Given a set of probabilities, normalize them to have a desired mean.
    
    Args:
        probs (torch.Tensor): The probabilities to normalize. Must be in the range [0, 1].
        mean (float): The desired mean of the probabilities.
    
    Returns:
        torch.Tensor: The normalized probabilities. Same dimension as the input probs.
    """
    #make sure negative sparsity levels lead to all zeros in the weights
    if mean <= 0:
        return 0. * probs
    
    mu = torch.mean(probs)
    
    if mu >= mean:
        return (mean / mu) * probs
    else:
        return 1 - (1 - mean)/(1 - mu) * (1 - probs)
    
def bernouli_gumbel_sample(probs, tau):
    """
    Draw hard samples from a given set of bernoulli probabilities using the Gumbel-Softmax trick.
    
    Args:
        probs (torch.Tensor): The probabilities to sample from.
        tau (float): The temperature parameter for the Gumbel-Softmax distribution.
    
    Returns:
        torch.Tensor: The hard samples. Same dimension as the input probs.
    """
    if torch.count_nonzero(probs).item() == 0:
        return 0. * probs
    
    #Sampling requires us to draw a gumbel sample for each category/binary outcome
    prob_01 = torch.stack((1. - probs, probs), dim=0) #[2, (dims)]

    #pytorch function requires un-normalized log-probabilities
    gumbel_sample = F.gumbel_softmax(torch.log(prob_01), tau=tau, hard=True, dim=0)[1] #[(dims)]

    return gumbel_sample

def bernouli_straight_through_sample(probs):
    """
    Draw hard samples from a given set of bernoulli probabilities using the Straight-Through trick.
    
    Args:
        probs (torch.Tensor): The probabilities to sample from.
        
    Returns:
        torch.Tensor: The hard samples. Same dimension as the input probs.
    """
    if torch.count_nonzero(probs).item() == 0:
        return 0. * probs
    
    random_samples = torch.rand_like(probs) #values in [0, 1]
    hard_samples = torch.zeros_like(probs)
    hard_samples[random_samples < probs] = 1.
    
    #propagate values of hard samples and gradients of probs
    return hard_samples.detach() + (probs - probs.detach()) 

def get_random_logits(length, dist, normalize=False, norm_mean=0):
    """
    Generates a set of random logits for a binary decision.
    
    Args:
        length (int): The number of logits to generate.
        dist (str): The distribution to sample from. Options in [normal, uniform, random].
        normalize (bool): Whether to normalize the logits to have a mean of norm_mean in probability. 
                          Defaults to False.
        norm_mean (float): The desired mean of the probabilities. Only used if normalize is True.
                           Defaults to 0.
    
    Returns:
        torch.Tensor: The generated logits. Shape [length].
    """
    if dist == "normal":
        logits = torch.randn(length)
    else:
        if dist == "uniform":
            probs = torch.ones(length) * 0.5
        elif dist == "random":
            probs = torch.rand(length)
            
        logits = torch.special.logit(probs, eps=1e-3)
    
    if normalize:
        probs = torch.sigmoid(logits)
        normed_probs = normalize_probs(probs=probs, mean=norm_mean)
        logits = torch.special.logit(normed_probs, eps=1e-3)
    
    return logits

def get_furthest_point_3d(query_point_inds, key_point_inds, grid_x, grid_y, include_conjugate_keys=False):
    """
    Given a list of query point indexes and key point indexes, find the index of the query point
        that is furthest from its nearest key point.
    Uses the L2 distance between points to determine the furthest point.
        
    Args:
        query_point_inds (array-like of ints): The indexes of the query points. Shape [m].
        key_point_inds (array-like of ints): The indexes of the key points. Shape [n].
        grid_x (torch.Tensor): The x-coordinates of the grid. Shape [length, length].
        grid_y (torch.Tensor): The y-coordinates of the grid. Shape [length, length].
        include_conjugate_keys (bool): Whether to include the conjugate version of the key points when calculating distances.
                                       If True, will consider a copy of each key point rotated 180 degrees about the origin. 
                                       Defaults to False. 
    Returns:
        int: The index of the furthest query point.
    """
    xy_grid = torch.stack([grid_x, grid_y], dim=0)
    xy_grid = torch.flatten(xy_grid, start_dim=1) #[2, length^2] grid of (x, y) points
    
    query_xy = xy_grid[:, query_point_inds] #[2, |query_point_inds|] array of (x, y) coords of query points
    key_xy = xy_grid[:, key_point_inds] #[2, |key_point_inds|] array of (x, y) coords of key points
    if include_conjugate_keys:
        key_xy = torch.cat((key_xy, -key_xy), dim=1) #[2, 2*|key_point_inds|] array of (x, y) coords of key points and their conjugates
    
    diff_xy = query_xy[..., None] - key_xy[:, None, ...] #[2, |query_point_inds|, |key_point_inds|] 2D differences
    squared_dists = torch.sum(torch.square(diff_xy), dim=0) #[|query_point_inds|, |key_point_inds|]  squared distances
    
    row_mins = torch.min(squared_dists, dim=1)[0] #[|query_point_inds|] array of distance to closest key point for each query point
    out_idx = torch.argmax(row_mins)
    
    return out_idx

def shape_mask_3d(length, flat_input, flat_input_idx, on_idx, off_idx):
    """
    Shapes a given flat set of weights into a 3D mask.
    
    Args:
        length (int): The length of the 3D mask.
        flat_input (torch.Tensor): The flat input weights. Shape [n, m] or [m].
        flat_input_idx (np.ndarray): The indexes of the input weights in the flattened final mask. Shape [m].
        on_idx (np.ndarray): Indexes of pattern entries to set to 1.
        off_idx (np.ndarray): Indexes of pattern entries to set to 0.
    
    Returns:
        torch.Tensor: The shaped 3D mask. Shape [n, length, length] (n=1 if flat_input is 1d).
    """
    if flat_input.dim() == 1:
        flat_input = flat_input.unsqueeze(0)
    n = flat_input.shape[0]
    
    flat_output = torch.zeros((n, length**2), device=flat_input.device, dtype=flat_input.dtype)
    flat_output[:, flat_input_idx] = flat_input
    flat_output[:, off_idx] = 0.
    flat_output[:, on_idx] = 1.
    
    return flat_output.view(n, length, length)
