import torch
import torch.fft as torch_fft

def ifft(x):
    """
    Centered, orthogonal ifft in torch. 
    
    Args:
        x (torch.Tensor): The input tensor. Dimension should be [N, C, H, W] complex.
        
    Returns:
        torch.Tensor: The ifft of x.
    """
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

def fft(x):
    """
    Centered, orthogonal fft in torch. 
    
    Args:
        x (torch.Tensor): The input tensor. Dimension should be [N, C, H, W] complex.
    
    Returns:
        torch.Tensor: The fft of x.
    """
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def normalize(x, x_min, x_max):
    """
    Scales x to appx [-1, 1].
    
    Args:
        x (torch.Tensor): The input tensor.
        x_min (float): The minimum value of x.
        x_max (float): The maximum value of x.
    
    Returns:
        torch.Tensor: The normalized tensor.
    """
    out = (x - x_min) / (x_max - x_min)
    return 2*out - 1

def unnormalize(x, x_min, x_max):
    """
    Takes input in appx [-1,1] and unscales it.
    
    Args:
        x (torch.Tensor): The input tensor.
        x_min (float): The minimum value of x.
        x_max (float): The maximum value of x.
    
    Returns:
        torch.Tensor: The unnormalized tensor.
    """
    out = (x + 1) / 2
    return out * (x_max - x_min) + x_min

def real_to_complex(x):
    """
    Takes an [N, 2, H, W] real-valued tensor and converts it to a [N, H, W] complex tensor.
    
    Args:
        x (torch.Tensor): The input tensor. [N, 2, H, W] real-valued.
    
    Returns:
        torch.Tensor: The converted tensor. [N, H, W] complex.
    """
    return torch.complex(x[:,0], x[:,1])

def complex_to_real(x):
    """
    Converts [N, H, W] complex tensor to a [N, 2, H, W] real-valued tensor.
    
    Args:
        x (torch.Tensor): The input tensor. [N, H, W] complex.
    
    Returns:
        torch.Tensor: The converted tensor. [N, 2, H, W] real-valued.
    """
    return torch.permute(torch.view_as_real(x), (0, 3, 1, 2))

def get_mvue_torch(y, s_maps):
    """
    Given multi-coil measurements and coil sensitivity maps, return the MVUE.
    
    Args:
        y (torch.Tensor): Undersampled multi-coil measurements PFSx. [N, C, H, W] complex tensor.
        s_maps (torch.Tensor): Coil sensitivity maps S. [N, C, H, W] complex tensor. 
    
    Returns:
        torch.Tensor: The estimated MVUE. [N, 2, H, W] real-valued tensor.
    """
    estimated_mvue = torch.sum(ifft(y) * torch.conj(s_maps), axis=1) / torch.sqrt(torch.sum(torch.square(torch.abs(s_maps)), axis=1))
    
    return complex_to_real(estimated_mvue)

def get_min_max(x):
    """
    Given a [N, 2, H, W] real-valued tensor, return [N, 1, 1, 1] shaped
        minimum and maximum pixel values.
        
    Args:
        x (torch.Tensor): The input tensor. [N, 2, H, W] real-valued.
    
    Returns:
        tuple: The minimum and maximum pixel values. (torch.Tensor, torch.Tensor) each [N, 1, 1, 1] real-valued.
    """
    norm_mins = torch.amin(x, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
    norm_maxes = torch.amax(x, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
    
    return norm_mins, norm_maxes

def MRI_forward_nomask(x, s_maps):
    """
    Forward model for MRI without a mask, i.e. FSx.
    
    Args:
        x (torch.Tensor): The input tensor. [N, 2, H, W] real-valued or [N, H, W] complex.
        s_maps (torch.Tensor): The coil sensitivity maps. [N, C, H, W] complex.
    
    Returns:
        torch.Tensor: The k-space data. [N, C, H, W] complex.
    """
    x_complex = x if torch.is_complex(x) else torch.complex(x[:,0], x[:,1]) # [N, H, W] complex

    coils = x_complex[:, None] * s_maps # Broadcast pointwise multiply

    return fft(coils) # Convert to k-space data

def MRI_forward(x, s_maps, mask):
    """
    Forward model for MRI with a mask, i.e. PFSx.
    
    Args:
        x (torch.Tensor): The input tensor. [N, 2, H, W] real-valued or [N, H, W] complex.
        s_maps (torch.Tensor): The coil sensitivity maps. [N, C, H, W] complex.
        mask (torch.Tensor): The mask to apply to the k-space data. [H, W] or [N, H, W] or [N, 1, H, W] real-valued.
    
    Returns:
        torch.Tensor: The k-space data. [N, C, H, W] complex.
    """
    ksp_coils = MRI_forward_nomask(x, s_maps) # [N, C, H, W] complex

    if len(mask.shape) == 2:
        mask_shaped = mask[None, None]
    elif len(mask.shape) == 3:
        mask_shaped = mask[:, None]
    
    return ksp_coils * mask_shaped # Apply mask to k-space data

def MRI_adjoint_nomask(y, s_maps):
    """
    Adjoint model for MRI without a mask, i.e.  (S*F*)y.
    
    Args:
        y (torch.Tensor): The input tensor in k-space. [N, C, H, W] complex.
        s_maps (torch.Tensor): The coil sensitivity maps S. [N, C, H, W] complex.
    
    Returns:
        torch.Tensor: The image data. [N, 2, H, W] real-valued.
    """
    pix_coils = ifft(y) #[N, C, H, W] complex

    x_raw = pix_coils * torch.conj(s_maps) #[N, C, H, W] complex

    x = torch.sum(x_raw, axis=1) #[N, H, W] complex

    return complex_to_real(x)

def hard_consistency(x, P, S, FSx):
    """
    Hard consistency operator for MRI, i.e. Adjoint((1-P)FSx^ + PFSx).
    
    Args:
        x (torch.Tensor): The input tensor, unnormalized. [N, 2, H, W] real-valued.
        P (torch.Tensor): The mask. [N, 1, H, W] real-valued.
        S (torch.Tensor): The coil sensitivity maps. [N, C, H, W] complex.
        FSx (torch.Tensor): The fully-sampled k-space data. [N, C, H, W] complex.
    
    Returns:
        torch.Tensor: The estimated image projected onto known measurements. [N, 2, H, W] real-valued.
    """
    y_repaint = (1 - P) * MRI_forward_nomask(x, S) + P * FSx
    x_repaint = get_mvue_torch(y_repaint, S)
    
    return x_repaint

