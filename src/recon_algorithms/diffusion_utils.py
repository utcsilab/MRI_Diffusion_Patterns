from src.recon_algorithms.edm import dnnlib
from src.utils.cg_utils import ZConjGrad, get_Aop_fun, get_cg_rhs
from src.utils.helpers import MRI_forward_nomask, complex_to_real, real_to_complex, hard_consistency

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

def MRI_diffusion_sampling(net, x_init, t_steps, FSx, P, S, alg_type, hard_consistent_output=False,
                           S_churn=0., S_min=0., S_max=float('inf'), S_noise=1.,
                           **kwargs):
    """
    Performs conditional sampling for solving MRI inverse problems using a diffusion-based model.
    
    Args:
        net (torch.nn.Module): The EDM diffusion network.
        x_init (torch.Tensor): The initial image. Should be normalized and noised. [N, 2, H, W] real-valued.
        t_steps (torch.Tensor): The noise schedule. [steps + 1] float.
        FSx (torch.Tensor): The fully-sampled k-space data. [N, C, H, W] complex-valued.
        P (torch.Tensor): The sampling pattern. [N, 1, H, W] real-valued.
        S (torch.Tensor): The coil sensitivity maps. [N, C, H, W] complex-valued.
        alg_type (str): The algorithm type for sampling. Value should be one of ["dps", "shallow_dps", "cg", "repaint"].
        hard_consistent_output (bool): Whether to enforce hard consistency in the final output. Defaults to False.
        S_churn (float): EDM randomness parameter.
        S_min (float): EDM sampling parameter.
        S_max (float): EDM sampling parameter.
        S_noise (float): EDM sampling parameter.
        **kwargs: Additional arguments for specific posterior sampling algorithms.
    
    Returns:
        torch.Tensor: The sampled image. [N, 2, H, W] real-valued.
    """
    #(0) Setup
    device = x_init.device
    
    FS = lambda x: MRI_forward_nomask(x, S) #[N, 2, H, W] float --> [N, C, H, W] complex
    
    class_labels = None
    if net.label_dim:
        class_labels = torch.zeros((x_init.shape[0], net.label_dim), device=device) #[N, label_dim]
    
    #(1) Sampling Loop
    x_next = x_init
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., T-1
        x_cur = x_next
        
        # (1a) Increase noise temporarily (for non-DDIM sampling).
        gamma = min(S_churn / (t_steps.numel() - 1), np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0 
        t_hat = net.round_sigma(t_cur + gamma * t_cur) 
        x_t_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur) 
        if alg_type == "dps":
            x_t_hat.requires_grad_()
        
        # (1b) Get the denoised network output
        x_denoised = net(x_t_hat, t_hat, class_labels)
        if alg_type == "shallow_dps":
            x_denoised.requires_grad_()
        
        # (1c) grab the negative score term d_cur
        if "dps" in alg_type:
            d_cur = (x_t_hat - x_denoised) / t_hat
        elif alg_type == "cg":
            x_cg_init = real_to_complex(x_denoised) #[N, 2, H, W] real --> [N, H, W] complex
            Aop_fun = get_Aop_fun(P, S)
            cg_rhs = get_cg_rhs(P, S, FSx, kwargs['cg_lambda'], x_cg_init)
            
            CG_Runner = ZConjGrad(cg_rhs, Aop_fun, kwargs['cg_max_iter'], kwargs['cg_lambda'], kwargs['cg_eps'], False)
            x_cg = CG_Runner.forward(x_cg_init)
            
            x_cg_real = complex_to_real(x_cg)
            
            d_cur = (x_t_hat - x_cg_real) / t_hat
        elif alg_type == "repaint":
            x_repaint = hard_consistency(x=x_denoised, P=P, S=S, FSx=FSx)
            
            d_cur = (x_t_hat - x_repaint) / t_hat
        else:
            raise NotImplementedError("Given alg_type not supported!")
        
        # (1d) grab the likelihood score
        if "dps" in alg_type:
            residual = P * (FS(x_denoised) - FSx)
            sse_per_samp = torch.sum(torch.square(torch.abs(residual)), dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
            sse = torch.sum(sse_per_samp)
            
            if alg_type == "shallow_dps":
                likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_denoised)[0] 
            else:
                likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_t_hat)[0] 
            
            if kwargs['normalize_grad']:
                likelihood_score = (kwargs['likelihood_step_size'] / torch.sqrt(sse_per_samp).detach()) * likelihood_score 
            else:
                likelihood_score = kwargs['likelihood_step_size'] * likelihood_score
        else:
            likelihood_score = 0. 
        
        # (1e) Take an Euler step using the gradient d_cur and the likelihood score
        x_next = x_t_hat + (t_next - t_hat) * d_cur - likelihood_score
        x_next = x_next.detach()
    
    x_hat = x_next
    if hard_consistent_output:
        x_hat = hard_consistency(x=x_hat, P=P, S=S, FSx=FSx) 
    
    return x_hat
