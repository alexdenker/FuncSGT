import numpy as np
import torch


class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError

class WhiteNoiseSampler(NoiseSampler):
    def __init__(self, n, device='cuda'):
        self.n = n
        self.device = device

    @torch.no_grad()
    def sample(self, N):
        return torch.randn(N, 1, self.n, device=self.device)

    def apply_C(self, x):
        return x
    
    def apply_Csqrt(self, x):
        return x

    def apply_Csqrt_inv(self, x):
        return x

import torch
import numpy as np

class SpectralNoiseSampler(NoiseSampler):
    def __init__(self, n, power=2.0, cutoff=None, device="cuda"):
        self.device = device
        self.n = n
        self.power = power 
        self.cutoff = cutoff
        # Pre-compute the filter (C^{1/2} in the spectral domain)
        self.spectral_filter = self.make_spectral_filter(n, power=power, cutoff=cutoff, device=device)

    def make_spectral_filter(self, n, power=2.0, cutoff=None, device=None):
        if device is None:
            device = self.device
        
        # d=1.0/n ensures frequencies are in 'physical' units relative to a domain of length 1
        k = torch.fft.fftfreq(n, d=1.0/n, device=device)  
        K2 = k**2
        
        # Decay ~ (1 + |k|^2)^{-power/2}
        filt = (1.0 + K2)**(-power/2.0)
        filt = filt.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, n)

        if cutoff is not None:
            filt = torch.fft.fftshift(filt, dim=(-1,))
            center = n // 2
            X = torch.arange(n, device=device)
            mask = (torch.abs(X - center) <= cutoff).float()
            filt = filt * mask.unsqueeze(0).unsqueeze(0)
            filt = torch.fft.ifftshift(filt, dim=(-1,))

        return filt

    @torch.no_grad()
    def sample(self, N):
        """
        Samples noise x = C^{1/2} z, where z ~ N(0, I).
        We scale by sqrt(n) so that the variance per point is resolution-invariant.
        """
        z = torch.randn(N, 1, self.n, device=self.device)
        return self.apply_Csqrt(z)

    def apply_Csqrt(self, x):
        """
        Computes y = C^{1/2} x. 
        The sqrt(n) scaling maintains consistent amplitude across resolutions.
        """
        # 1. Scale input to keep variance constant across grid densities
        x_scaled = x * np.sqrt(self.n)
        
        # 2. Transform to Fourier domain with ortho norm
        x_f = torch.fft.fft(x_scaled, norm="ortho")
        
        # 3. Apply the spectral filter
        y_f = x_f * self.spectral_filter
        
        # 4. Transform back
        return torch.fft.ifft(y_f, norm="ortho").real

    def apply_C(self, x):
        """
        Computes y = C x = (C^{1/2} * C^{1/2}) x.
        Note: The sqrt(n) factor is applied twice (once per C^{1/2}).
        """
        x_scaled = x * self.n # sqrt(n) * sqrt(n)
        x_f = torch.fft.fft(x_scaled, norm="ortho")
        y_f = x_f * (self.spectral_filter**2)
        return torch.fft.ifft(y_f, norm="ortho").real

    def apply_Csqrt_inv(self, x):
        """
        Computes y = (C^{1/2})^{-1} x.
        This is used for whitening or calculating the log-likelihood/score.
        """
        x_f = torch.fft.fft(x, norm="ortho")
        
        # Inverse the filter and the scaling
        y_f = x_f / (self.spectral_filter + 1e-8)
        y = torch.fft.ifft(y_f, norm="ortho").real
        
        return y / np.sqrt(self.n)


# class SpectralNoiseSampler(NoiseSampler):
#     def __init__(self, n, power=2.0, cutoff=None, device='cuda'):
#         self.device = device   
#         self.n = n

#         self.power = power 
#         self.cutoff = cutoff
#         self.spectral_filter = self.make_spectral_filter(n, power=power, cutoff=cutoff, device=device)

#     def make_spectral_filter(self, n, power=2.0, cutoff=None, device=None):
#         """Return real spectral filter of shape (1,1,n) to multiply FFT(z).
#         power controls decay ~ (1+|k|^2)^{-power/2}. cutoff (int) optionally zeroes high freq beyond cutoff.
#         """
#         if device is None:
#             device = self.device
        
#         k = torch.fft.fftfreq(n, d=1.0/n, device=device)  
#         K2 = k**2
        
#         # power/2.0, because we are defining C^{1/2}, as for sampling we want x = C^{1/2} w, w ~ N(0,I)
#         filt = (1.0 + K2)**(-power/2.0)

#         # put in (1,1,n)
#         filt = torch.tensor(filt, device=device).unsqueeze(0).unsqueeze(0)
#         if cutoff is not None:
#             filt = torch.fft.fftshift(filt, dim=(-1,))  # shift zero freq to center

#             # zero out frequencies with index > cutoff
#             center = n // 2
#             X = torch.arange(n, device=device)
#             R = torch.abs(X - center)
#             mask = (R <= cutoff).float()
#             filt = filt * mask.unsqueeze(0).unsqueeze(0)
#             filt = torch.fft.ifftshift(filt, dim=(-1,))  # shift back

#         filt = filt #/ torch.norm(filt) * np.sqrt(n)  
#         return filt

#     @torch.no_grad()
#     def sample(self, N):
#         z = torch.randn(N, 1, self.n, device=self.device) * np.sqrt(self.n)
#         z_f = torch.fft.fft(z, norm='ortho')
#         z_f = z_f * self.spectral_filter 
#         z = torch.fft.ifft(z_f, norm='ortho').real
#         return z

#     def apply_C(self, x):
#         z_f = torch.fft.fft(x, norm='ortho')
#         z_f = z_f * self.spectral_filter**2
#         z = torch.fft.ifft(z_f, norm='ortho').real
#         return z
    
#     def apply_Csqrt(self, x):
#         z_f = torch.fft.fft(x, norm='ortho')
#         z_f = z_f * self.spectral_filter
#         z = torch.fft.ifft(z_f, norm='ortho').real
#         return z

#     def apply_Csqrt_inv(self, x):
#         z_f = torch.fft.fft(x, norm='ortho')
#         z_f = z_f / (self.spectral_filter + 1e-6)
#         z = torch.fft.ifft(z_f, norm='ortho').real
#         return z

if __name__ == "__main__":
    n = 100
    batch_size = 8
    sampler = SpectralNoiseSampler(n=100, power=2.0, device="cpu")
    samples = sampler.sample(batch_size)
    print(samples.shape)  # should be (batch_size, 1, n)
    assert samples.shape == (
        batch_size,
        1,
        n,
    ), "Shape mismatch for spectral noise sampler"

    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(batch_size):
        plt.plot(samples[i, 0, :].cpu().numpy())
    plt.title("Spectral Noise Samples")
    plt.savefig("spectral_noise_samples.png")
    plt.show()

    batch_size = 4
    # test spectral noise at different resolutions 
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for n_points in [32, 64, 128]:
        sampler = SpectralNoiseSampler(n=n_points, power=2.0, device="cpu")
        samples = sampler.sample(batch_size)
        print(f"n_points: {n_points}, samples shape: {samples.shape}")
        assert samples.shape == (batch_size, 1, n_points), f"Shape mismatch for spectral noise sampler with n_points={n_points}"

        for i in range(batch_size):
            ax = axes[[32, 64, 128].index(n_points), i]
            ax.plot(samples[i,0,:].cpu().numpy())
            ax.set_title(f"n_points={n_points}")
    plt.tight_layout()
    plt.savefig('spectral_noise_samples_multi_resolution.png')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, n in enumerate([32, 64, 128]):
        sampler = SpectralNoiseSampler(n=n, power=2.0, device="cpu")
        samples = sampler.sample(100)
        for i in range(100):
            ax = axes[idx]
            ax.plot(samples[i,0,:].cpu().numpy())
        ax.set_title(f"n_points={n}")

    plt.tight_layout()
    plt.savefig('spectral_noise_samples_multi_resolution_2.png')
    plt.show()