
import torch


def fourier_coefficients(array, num_bases):
    """Array of shape [..., pts, dim]
    Returns array of shape [..., 2, :num_bases, dim]"""

    complex_coefficients = torch.fft.rfft(array, norm="forward", axis=-2)
    complex_coefficients = complex_coefficients[..., :num_bases, :]
    coeffs = torch.stack([complex_coefficients.real, complex_coefficients.imag], axis=1)
    return coeffs

def inverse_fourier(coefficients, num_pts):
    """Array of shape [..., 2, num_bases, dim]
    Returns array of shape [..., num_pts, dim]"""
    coeffs_real = coefficients[..., 0, :, :]
    coeffs_im = coefficients[..., 1, :, :]
    complex_coefficients = coeffs_real + 1j * coeffs_im
    return torch.fft.irfft(complex_coefficients, norm="forward", n=num_pts, axis=-2)


def get_fourier_noise_scales(num_bases, scale_type="inv_k_sq", device="cpu", dtype=torch.float32):
    """
    Compute noise scaling factors for each Fourier basis.
    
    For infinite-dimensional problems, noise should decay with frequency to maintain
    regularity. This function provides proper whitening of the noise covariance.
    
    Args:
        num_bases: Number of Fourier bases (e.g., 16)
        scale_type: Type of scaling
            - "inv_k_sq": 1/(k+1)^2 (Sobolev H^{-1}; recommended for shape data)
            - "inv_k": 1/(k+1) (weaker regularity)
            - "uniform": constant (standard i.i.d. noise; not recommended)
        device: torch device
        dtype: torch data type
    
    Returns:
        Tensor of shape [num_bases] with noise scaling factors
    """
    k = torch.arange(num_bases, dtype=dtype, device=device)
    
    if scale_type == "inv_k_sq":
        # 1/(k+1)^2 scaling - good for C^{1/2} regularity
        scales = 1.0 / (k + 1.0) ** 2.0
    elif scale_type == "inv_k":
        # 1/(k+1) scaling - weaker regularity
        scales = 1.0 / (k + 1.0)
    elif scale_type == "uniform":
        # Uniform scaling (i.i.d. noise)
        scales = torch.ones(num_bases, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")
    
    # Normalize so that the mean is 1 (for numerical stability)
    scales = scales / scales.mean()
    
    return scales

