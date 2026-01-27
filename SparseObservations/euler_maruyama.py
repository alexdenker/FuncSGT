
import torch 
from tqdm import tqdm 
from .score_model import ScoreModel

class EulerMaruyamaExponential():
    """
    Make use of an exponential integrator

    """
    def __init__(self, model: ScoreModel):
        self.model = model

    def sample(self, ts, pos, **kwargs):
        """
        ts: timesteps, [T] (forward in time)
        """
        batch_size = kwargs.get("batch_size", 1)
        verbose = kwargs.get("verbose", False)

        delta_t = ts[1] - ts[0]
        xt = self.model.noise_sampler.sample(batch_size)# N(0,C)
        pos_inp = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
        for ti in tqdm(reversed(ts), total=len(ts), disable=not verbose):
            t = torch.ones(batch_size).to(xt.device)* ti

            with torch.no_grad():
                score, _ = self.model(xt, t, grid=pos_inp)

            beta_t = self.model.sde.beta_t(t).view(-1, 1, 1, 1)
            noise = self.model.noise_sampler.sample(batch_size) # N(0,C)

            phi_t = torch.exp(1/2*beta_t*delta_t)
            xt = phi_t*xt + 2*(phi_t - 1) * score + (phi_t**2 - 1).sqrt() * noise

        return xt
    
class EulerMaruyama():
    def __init__(self, model: ScoreModel):
        self.model = model

    def sample(self, ts, pos, **kwargs):
        """
        ts: timesteps, [T] (forward in time)
        """
        batch_size = kwargs.get("batch_size", 1)
        verbose = kwargs.get("verbose", False)

        delta_t = ts[1] - ts[0]
        xt = self.model.noise_sampler.sample(batch_size) # N(0,C)
        pos_inp = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
        for ti in tqdm(reversed(ts), total=len(ts), disable=not verbose):
            t = torch.ones(batch_size).to(xt.device)* ti

            with torch.no_grad():
                score, _ = self.model(xt, t, grid=pos_inp)

            beta_t = self.model.sde.beta_t(t).view(-1, 1, 1, 1)
            noise = self.model.noise_sampler.sample(batch_size) # N(0,C)

            xt = xt + (beta_t/2.0 * xt + beta_t * score)*delta_t + beta_t.sqrt()*delta_t.sqrt() * noise

        return xt
