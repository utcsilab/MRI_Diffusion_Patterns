from src.recon_algorithms.diffusion_utils import get_noise_schedule, MRI_diffusion_sampling

class Diffusion_CG:
    def __init__(self, net, cg_lambda, cg_max_iter, cg_eps, hard_consistent_output,
                 steps, rho, S_churn, S_min, S_max, S_noise, sigma_min, sigma_max):
        """
        Class that performs diffusion with conjugate gradient for MRI reconstruction.
        
        Args:
            net (torch.nn.Module): The EDM diffusion network.
            cg_lambda (float): The CG regularization parameter.
            cg_max_iter (int): The maximum number of CG iterations.
            cg_eps (float): The CG stopping criterion.
            hard_consistent_output (bool): Whether to enforce hard consistency in the final output.
            steps (int): The number of steps in the reverse diffusion process.
            rho (float): The noise schedule parameter in EDM.
            S_churn (float): EDM randomness parameter.
            S_min (float): EDM sampling parameter.
            S_max (float): EDM sampling parameter.
            S_noise (float): EDM sampling parameter.
            sigma_min (float): The minimum/ending noise standard deviation.
            sigma_max (float): The maximum/starting noise standard deviation.
        """
        self.net = net
        
        self.cg_lambda = cg_lambda
        self.cg_max_iter = cg_max_iter
        self.cg_eps = cg_eps
        self.hard_consistent_output = hard_consistent_output
        
        self.steps = steps
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def __call__(self, x_init, FSx, P, S):
        """
        Performs performs diffusion with conjugate gradient for MRI reconstruction.
        
        Args:
            x_init (torch.Tensor): The initial image. Should be normalized and noised. [N, 2, H, W] real-valued.
            FSx (torch.Tensor): The fully-sampled k-space data. [N, C, H, W] complex-valued.
            P (torch.Tensor): The sampling pattern. [N, 1, H, W] real-valued.
            S (torch.Tensor): The coil sensitivity maps. [N, C, H, W] complex-valued.
        
        Returns:
            torch.Tensor: The reconstructed image. [N, 2, H, W] real-valued.
        """
        # Get noise schedule.
        t_steps = get_noise_schedule(steps=self.steps, sigma_max=self.sigma_max, sigma_min=self.sigma_min, 
                                     rho=self.rho, net=self.net, device=x_init.device)
        
        # Perform MRI diffusion sampling.
        x_out = MRI_diffusion_sampling(net=self.net, x_init=x_init, t_steps=t_steps, FSx=FSx, P=P, S=S,
                                       alg_type="dps", hard_consistent_output=self.hard_consistent_output,
                                       S_churn=self.S_churn, S_min=self.S_min, S_max=self.S_max, S_noise=self.S_noise,
                                       cg_lambda=self.cg_lambda, cg_max_iter=self.cg_max_iter, cg_eps=self.cg_eps)
        
        return x_out
    