import torch


#Implementation of the forward SDE
#dXt = -1/2 * beta(t) Xt dt + \sqrt{beta(t) * C} dWt
class OU:
    def __init__(self, beta_min=0.001, beta_max=20):
        
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        return self.beta_min + t*(self.beta_max - self.beta_min)
    
    def drift(self, t, x):

        if len(t.shape) == 1:
            t = t.view(-1, *([1] * (x.ndim - 1)))
        return -0.5*self.beta_t(t)*x
    
    def diffusion(self, t, x):

        if len(t.shape) == 1:
            t = t.view(-1, *([1] * (x.ndim - 1)))
        return self.beta_t(t).sqrt()

    def alpha_t(self, t):
        """
        int_0^t beta(s) ds = t * beta_min + 1/2 * t**2 (beta_max - beta_min)
        
        """

        return t*self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)
    
    def mean_t(self, t, x):
        if len(t.shape) == 1:
            t = t.view(-1, *([1] * (x.ndim - 1)))

        return torch.exp(-1/2*self.alpha_t(t))*x

    def mean_t_scaling(self, t, x):
        if len(t.shape) == 1:
            t = t.view(-1, *([1] * (x.ndim - 1)))

        return torch.exp(-1/2*self.alpha_t(t))

    def std_t_scaling(self, t, x):
        if len(t.shape) == 1:
            t = t.view(-1, *([1] * (x.ndim - 1)))

        return (1 - torch.exp(-self.alpha_t(t))).sqrt()

if __name__ == "__main__":

    sde = OU()

    x = torch.randn((6, 1, 100))
    t = torch.randn((x.shape[0],))

    d = sde.drift(t, x)

    print(d.shape)

    mean_t = sde.mean_t(t, x)

    print(mean_t.shape)