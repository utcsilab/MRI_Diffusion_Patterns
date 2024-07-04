import torch
from torchmetrics.functional import structural_similarity_index_measure
from src.utils.helpers import MRI_forward_nomask

def cosine_similarity_loss(x_hat, x):
    x_hat_norm = torch.norm(x_hat, p=2, dim=(1,2,3))
    x_norm = torch.norm(x, p=2, dim=(1,2,3))
    cosine_sim_per_sample = torch.sum(x_hat * x, dim=(1,2,3)) / (x_hat_norm * x_norm)
    
    return torch.mean(1 - cosine_sim_per_sample)

def calculate_loss(x_hat, x, loss_type="l2"):
    """
    Calculates loss between a reconstrution x_hat and true x.
    
    Args:
        x_hat (torch.Tensor): The reconstruction. [N, 2, H, W] real-valued.
        x (torch.Tensor): The true image. [N, 2, H, W] real-valued
        loss_type (str): The type of loss to calculate. Should be in ['l2', 'l1', 'ssim'].
    
    Returns:
        torch.Tensor: The loss, calculated as sample-wise mean of individual losses.
    """
    if loss_type == "l2":
        #Sample-Wise Mean Normalised SSE
        numerator = torch.sum(torch.square(x_hat - x), dim=(1,2,3))
        denominator = torch.sum(torch.square(x), dim=(1,2,3))
        loss = torch.mean(numerator / denominator)
    elif loss_type == "l1":
        #Sample-Wise Mean Normalised SAE
        numerator = torch.sum(torch.abs(x_hat - x), dim=(1,2,3))
        denominator = torch.sum(torch.abs(x), dim=(1,2,3))
        loss = torch.mean(numerator / denominator)
    elif loss_type == "ssim":
        #Sample-wise Mean SSIM
        #Manually iterate instead of using batched solution since data_range argument of 
        #   metric does not accept per-sample pixel ranges   
        pred = torch.norm(x_hat, dim=1, keepdim=True) #[N,1,H,W] Magnitude image
        target = torch.norm(x, dim=1, keepdim=True)
        ssim_loss_list = []
        for i in range(target.shape[0]):
            #The double slice indexing [[i]] slices and keeps dimension intact
            ssim_loss_list.append((1 - structural_similarity_index_measure(preds=pred[[i]], 
                                                                            target=target[[i]], 
                                                                            reduction="sum")))
        loss = torch.mean(torch.stack(ssim_loss_list))
    else:
        raise NotImplementedError("LOSS NOT IMPLEMENTED!")
    
    return loss

def make_noisy_sample(x, sigma_t=None):
    """
    Adds random Gaussian noise to an input.

    Args:
        x (torch.Tensor): The clean image. [N, 2, H, W] real-valued.
        sigma_t (torch.Tensor): The noise standard deviation at time t. [N, 1, 1, 1] real-valued.
                                Default is None, in which case it is randomly sampled in (0, 1).
    
    Returns: 
        x_t (torch.Tensor): The noised sample. [N, 2, H, W] real-valued
        sigma_t (torch.Tensor): Noise standard deviation at time t. [N, 1, 1, 1] real-valued.
    """
    if sigma_t is None:
        sigma_t = torch.rand(1, device=x.device)
        sigma_t = sigma_t.expand(x.shape[0], 1, 1, 1)
        
    n = torch.randn_like(x) * sigma_t
    
    x_t = x + n
    
    return x_t, sigma_t

def single_step_posterior_estimate(net, x_t, sigma_t, FSx, P, S, likelihood_step_size):
    """
    Performs a single-step reconstruction using the posterior sampling version of Tweedie's formula.
    Uses the approximation: E[x_0 | x_t, y] = x^ - c * (d / dx^)||PFSx^ - y||^2, where x^ = E[x_0 | x_t].

    Args:
        net (torch.nn.Module): The diffusion network, assumed to be a denoiser (i.e. net(x + noise) = x).
                                Assumed to be an EDM network (Karras et al., 2022).
        x_t (torch.Tensor): The noised sample. [N, 2, H, W] real-valued. 
                            The second axis holds the real and imaginary components.
        sigma_t (torch.Tensor): Noise standard deviation at time t. [N, 1, 1, 1] float.
        FSx (torch.Tensor): Fully-sampled ground truth k-space with C coil measurements.
                            [N, C, H, W] complex-valued.
        P (torch.Tensor): The sampling pattern, entries should be in [0, 1].
                            [N, 1, H, W] real-valued.
        S (torch.Tensor): Sensitivity maps for each of C coils for each of the N samples.
                          [N, C, H, W] complex valued..
        likelihood_step_size (float): Step size parameter for the likelihood term.
    
    Returns:
        x_hat (torch.Tensor): Single-step posterior reconstruction E[x_0 | x_t, y].
                                [N, 2, H, W] real-valued.
        x_hat_0 (torch.Tensor): The denoised estimate E[x_0 | x_t].
                                 [N, 2, H, W] real-valued.                        
    """
    #(0) Setup
    device = x_t.device
    
    class_labels = None
    if net.label_dim:
        class_labels = torch.zeros((x_t.shape[0], net.label_dim), device=device) #[N, label_dim]
    
    #(1) Get the unconditional denoised estimate
    x_hat_0 = net(x_t, sigma_t, class_labels)
    x_hat_0.requires_grad_()
    
    #(2) Calculate the likelihood gradient
    residual = P * (MRI_forward_nomask(x_hat_0, S) - FSx)
    sse = torch.sum(torch.square(torch.abs(residual)))
    likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat_0, create_graph=True)[0] #create a graph to calculate loss gradients
    
    #(3) Create the final posterior mean prediction
    x_hat = x_hat_0 - likelihood_step_size * likelihood_score
    
    return x_hat, x_hat_0
