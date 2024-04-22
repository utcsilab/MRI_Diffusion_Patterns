from src.recon_algorithms.diffusion_utils import get_noise_schedule, MRI_diffusion_sampling

class DPS:
    def __init__(self, net, likelihood_step_size, normalize_grad, hard_consistent_output,
                 steps, rho, S_churn, S_min, S_max, S_noise, sigma_min, sigma_max):
        """
        Class that performs DPS for MRI reconstruction.
        
        Args:
            net (torch.nn.Module): The EDM diffusion network.
            likelihood_step_size (float): The step size for the likelihood term in DPS.
            normalize_grad (bool): Whether to normalize the likelihood gradient in DPS.
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
        
        self.likelihood_step_size = likelihood_step_size
        self.normalize_grad = normalize_grad
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
        Performs DPS for MRI reconstruction.
        
        Args:
            x_init (torch.Tensor): The initial image. Should be normalized and noised. [N, 2, H, W] real-valued.
            FSx (torch.Tensor): The fully-sampled k-space data. [N, C, H, W] complex-valued.
            P (torch.Tensor): The sampling pattern. [N, 1, H, W] real-valued.
            S (torch.Tensor): The coil sensitivity maps. [N, C, H, W] complex-valued.
        
        Returns:
            torch.Tensor: The reconstructed image. [N, 2, H, W] real-valued.
        """
        # Get noise schedule.
        t_steps = get_noise_schedule(self.steps, self.sigma_max, self.sigma_min, self.rho, self.net, x_init.device)
        
        # Perform MRI diffusion sampling.
        x_out = MRI_diffusion_sampling(self.net, x_init, t_steps, FSx, P, S, "dps", self.hard_consistent_output,
                                       self.S_churn, self.S_min, self.S_max, self.S_noise,
                                       normalize_grad=self.normalize_grad, likelihood_step_size=self.likelihood_step_size)
        
        return x_out
    