import torch 
import numpy as np 

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError

class SpectralNoiseSampler(NoiseSampler):
    def __init__(self, n, power=2.0, cutoff=None, device='cuda'):
        self.device = device   
        self.n = n

        self.power = power 
        self.cutoff = cutoff
        self.spectral_filter = self.make_spectral_filter(n, power=power, cutoff=cutoff, device=device)

    def make_spectral_filter(self, n, power=2.0, cutoff=None, device=None):
        """Return real spectral filter of shape (1,1,n) to multiply FFT(z).
        power controls decay ~ (1+|k|^2)^{-power/2}. cutoff (int) optionally zeroes high freq beyond cutoff.
        """
        if device is None:
            device = self.device
        
        k = torch.fft.fftfreq(n, d=1.0/n, device=device)  
        K2 = k**2
        
        # power/2.0, because we are defining C^{1/2}, as for sampling we want x = C^{1/2} w, w ~ N(0,I)
        filt = (1.0 + K2)**(-power/2.0)

        # put in (1,1,n)
        filt = torch.tensor(filt, device=device).unsqueeze(0).unsqueeze(0)
        if cutoff is not None:
            filt = torch.fft.fftshift(filt, dim=(-1,))  # shift zero freq to center

            # zero out frequencies with index > cutoff
            center = n // 2
            X = torch.arange(n, device=device)
            R = torch.abs(X - center)
            mask = (R <= cutoff).float()
            filt = filt * mask.unsqueeze(0).unsqueeze(0)
            filt = torch.fft.ifftshift(filt, dim=(-1,))  # shift back

        filt = filt #/ torch.norm(filt) * np.sqrt(n)  
        return filt

    @torch.no_grad()
    def sample(self, N):
        z = torch.randn(N, 1, self.n, device=self.device)
        z_f = torch.fft.fft(z)
        z_f = z_f * self.spectral_filter
        z = torch.fft.ifft(z_f).real
        return z

    def apply_C(self, x):
        z_f = torch.fft.fft(x)
        z_f = z_f * self.spectral_filter**2
        z = torch.fft.ifft(z_f).real
        return z
    
    def apply_Csqrt(self, x):
        z_f = torch.fft.fft(x)
        z_f = z_f * self.spectral_filter
        z = torch.fft.ifft(z_f).real
        return z

    def apply_Csqrt_inv(self, x):
        z_f = torch.fft.fft(x)
        z_f = z_f / (self.spectral_filter + 1e-6)
        z = torch.fft.ifft(z_f).real
        return z

if __name__ == "__main__":
    n = 100
    batch_size = 8
    sampler = SpectralNoiseSampler(n=100, power=2.0, device="cpu")
    samples = sampler.sample(batch_size)
    print(samples.shape)  # should be (batch_size, 1, n)
    assert samples.shape == (batch_size, 1, n), "Shape mismatch for spectral noise sampler"

    import matplotlib.pyplot as plt 

    plt.figure()
    for i in range(batch_size):
        plt.plot(samples[i,0,:].cpu().numpy())
    plt.title('Spectral Noise Samples')
    plt.savefig('spectral_noise_samples.png')
    plt.show()