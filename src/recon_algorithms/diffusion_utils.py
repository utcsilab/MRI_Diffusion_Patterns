from src.recon_algorithms.edm import dnnlib

import torch
import numpy as np
import pickle


def load_net(net_path, device):
    """
    Loads a network from a .pth file.

    Args:
        net_path (str): The path to the network file.
        device (torch.device): The device to load the network on.

    Returns:
        torch.nn.Module: The network.
    """
    with dnnlib.util.open_url(net_path) as f:
        net = pickle.load(f)['ema'].to(device)
    
    return net

def get_noise_schedule(steps, sigma_max, sigma_min, rho, net, device):
    """
    Generates a [steps + 1] torch tensor with sigma values for reverse diffusion
        in descending order (final entry is always 0).

    Args:
        steps (int): Number of steps in the reverse diffusion process.
        sigma_max (float): The maximum/starting noise standard deviation.
        sigma_min (float): The minimum/ending noise standard deviation.
        rho (float): The noise schedule parameter in EDM.  
        net (torch.nn.Module): The EDM diffusion network.
        device (torch.device): The device to generate the noise schedule on.
    
    Returns:
        torch.Tensor: The noise schedule. [steps + 1] float.
    """
    step_indices = torch.arange(steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    return t_steps

