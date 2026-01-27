import torch 


class ScoreModel:
    def __init__(self, model, sde, noise_sampler, model_type):
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler  
        self.model_type = model_type


    def __call__(self, x, t, grid):
        """
        x: [batch_size, 1, n_points]
        t: [batch_size]
        grid: [batch_size, 1, n_points]

        return:
        score: [batch_size, 1, n_points]
        x0hat: [batch_size, 1, n_points]
        """
        std_factor = self.sde.std_t_scaling(t, x)
        score = self.model(x, grid, t).unsqueeze(1) / std_factor  # shape: [batch_size, 1, n_points]
        if self.model_type == "raw":  # the network output is nabla log p
            score = self.noise_sampler.apply_C(score)
        elif self.model_type == "C_sqrt": # the network output is C^{1/2} nabla log p
            score = self.noise_sampler.apply_Csqrt(score)
        elif self.model_type == "C": # the network output is C nabla log p
            score = score 
        else:
            raise NotImplementedError

        mean_t_scale = self.sde.mean_t_scaling(t, x)

        x0hat = (x + std_factor**2 * score) / mean_t_scale

        return score, x0hat
    
class CondScoreModel:
    def __init__(self, model, sde, noise_sampler, model_type):
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler  
        self.model_type = model_type


    def __call__(self, x, y, t, grid):
        """
        x: [batch_size, 1, n_points]
        y: [batch_size, 1, n_points]
        t: [batch_size]
        grid: [batch_size, 1, n_points]

        return:
        score: [batch_size, 1, n_points]
        x0hat: [batch_size, 1, n_points]
        """
        std_factor = self.sde.std_t_scaling(t, x)
        score = self.model(x, y, grid, t).unsqueeze(1) / std_factor  # shape: [batch_size, 1, n_points]
        if self.model_type == "raw":  # the network output is nabla log p
            score = self.noise_sampler.apply_C(score)
        elif self.model_type == "C_sqrt": # the network output is C^{1/2} nabla log p
            score = self.noise_sampler.apply_Csqrt(score)
        elif self.model_type == "C": # the network output is C nabla log p
            score = score 
        else:
            raise NotImplementedError

        mean_t_scale = self.sde.mean_t_scaling(t, x)

        x0hat = (x + std_factor**2 * score) / mean_t_scale

        return score, x0hat


class HtransformModel:
    def __init__(self, h_trans, model, sde, noise_sampler, model_type, forward_op, cond_model_type):
        self.h_trans = h_trans
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler
        self.model_type = model_type
        self.forward_op = forward_op
        self.cond_model_type = cond_model_type

    def __call__(self, x, y, t, grid):
        std_t = self.sde.std_t_scaling(t, x)
        mean_t_scale = self.sde.mean_t_scaling(t, x)

        with torch.no_grad():
            pred = self.model(x, grid, t).unsqueeze(1) / std_t

            if self.model_type == "raw":  # the network output is nabla log p
                pred = self.noise_sampler.apply_C(pred)
            elif self.model_type == "C_sqrt": # the network output is C^{1/2} nabla log p
                pred = self.noise_sampler.apply_Csqrt(pred) 
            elif self.model_type == "C": # the network output is C nabla log p
                pred = pred 
            else:
                raise NotImplementedError

        x0hat = (x + std_t**2 * pred) / mean_t_scale

        with torch.enable_grad():
            x0hat.requires_grad_(True)
            y_pred = self.forward_op(x0hat.squeeze(1))
            loss_y = torch.mean((y_pred - y)**2)  # MSE loss
            log_likelihood_grad = torch.autograd.grad(loss_y.sum(), x0hat, retain_graph=True)[0].detach()
            #log_likelihood_grad = self.noise_sampler.apply_C(log_likelihood_grad)

        pred_h = self.h_trans(x, y, log_likelihood_grad, grid, t).unsqueeze(1) / std_t    
        if self.cond_model_type == "raw":  # the network output is nabla log p
            pred_h = self.noise_sampler.apply_C(pred_h)
        elif self.cond_model_type == "C_sqrt": # the network output is C^{1/2} nabla log p
            pred_h = self.noise_sampler.apply_Csqrt(pred_h) 
        elif self.cond_model_type == "C": # the network output is C nabla log p
            pred_h = pred_h 
        else:
            raise NotImplementedError

        pred = pred + pred_h

        return pred, None 

