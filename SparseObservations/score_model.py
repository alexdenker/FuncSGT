import torch

class ScoreModelFinDim:
    def __init__(self, model, sde, noise_sampler):
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler  

    def __call__(self, x, t):
        """
        x: [batch_size, 1, n_points]
        t: [batch_size]

        return:
        score: [batch_size, 1, n_points]
        x0hat: [batch_size, 1, n_points]
        """
        std_factor = self.sde.std_t_scaling(t, x)
        score = self.model(x, t).unsqueeze(1) / std_factor  # shape: [batch_size, 1, n_points]
       
        mean_t_scale = self.sde.mean_t_scaling(t, x)

        x0hat = (x + std_factor**2 * score) / mean_t_scale

        return score, x0hat
    

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
        score = (
            self.model(x, grid, t).unsqueeze(1) / std_factor
        )  # shape: [batch_size, 1, n_points]
        if self.model_type == "raw":  # the network output is nabla log p
            score = self.noise_sampler.apply_C(score)
        elif self.model_type == "C_sqrt":  # the network output is C^{1/2} nabla log p
            score = self.noise_sampler.apply_Csqrt(score)
        elif self.model_type == "C":  # the network output is C nabla log p
            score = score
        else:
            raise NotImplementedError

        mean_t_scale = self.sde.mean_t_scaling(t, x)

        x0hat = (x + std_factor**2 * score) / mean_t_scale

        return score, x0hat


class CondScoreModel:
    def __init__(self, model, sde, noise_sampler, model_type, cond_operator):
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler
        self.model_type = model_type
        self.cond_operator = cond_operator

    def __call__(self, x, y, t, grid):
        """
        x: [batch_size, 1, n_points]
        y: (eval_points, values)
        t: [batch_size]
        grid: [batch_size, 1, n_points]

        return:
        score: [batch_size, 1, n_points]
        x0hat: [batch_size, 1, n_points]
        """
        eval_points, values = y
        v, m = self.cond_operator(eval_points, values)  # v,m: [batch_size, n_points]
        v = v.unsqueeze(1)  # [batch_size, 1, n_points]
        m = m.unsqueeze(1)  # [batch_size, 1, n_points]
        y = torch.cat([v, m], dim=1)  # [batch_size, 2, n_points]

        std_factor = self.sde.std_t_scaling(t, x)
        score = (
            self.model(x, y, grid, t).unsqueeze(1) / std_factor
        )  # shape: [batch_size, 1, n_points]
        if self.model_type == "raw":  # the network output is nabla log p
            score = self.noise_sampler.apply_C(score)
        elif self.model_type == "C_sqrt":  # the network output is C^{1/2} nabla log p
            score = self.noise_sampler.apply_Csqrt(score)
        elif self.model_type == "C":  # the network output is C nabla log p
            score = score
        else:
            raise NotImplementedError

        mean_t_scale = self.sde.mean_t_scaling(t, x)

        x0hat = (x + std_factor**2 * score) / mean_t_scale

        return score, x0hat


class HtransformModel:
    def __init__(
        self,
        h_trans,
        model,
        sde,
        noise_sampler,
        model_type,
        cond_model_type,
        cond_operator,
    ):
        self.h_trans = h_trans
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler
        self.model_type = model_type
        self.cond_model_type = cond_model_type
        self.cond_operator = cond_operator

    def __call__(self, x, y, t, grid, forward_op):

        std_t = self.sde.std_t_scaling(t, x)
        mean_t_scale = self.sde.mean_t_scaling(t, x)

        eval_points, observations = y
        v, m = self.cond_operator(
            eval_points, observations
        )  # v,m: [batch_size, n_points]
        v = v.unsqueeze(1)  # [batch_size, 1, n_points]
        m = m.unsqueeze(1)  # [batch_size, 1, n_points]
        y_inp = torch.cat([v, m], dim=1)  # [batch_size, 2, n_points]

        with torch.no_grad():
            pred = self.model(x, grid, t).unsqueeze(1) / std_t

            if self.model_type == "raw":  # the network output is nabla log p
                pred = self.noise_sampler.apply_C(pred)
            elif (
                self.model_type == "C_sqrt"
            ):  # the network output is C^{1/2} nabla log p
                pred = self.noise_sampler.apply_Csqrt(pred)
            elif self.model_type == "C":  # the network output is C nabla log p
                pred = pred
            else:
                raise NotImplementedError

        x0hat = (x + std_t**2 * pred) / mean_t_scale

        with torch.enable_grad():
            x0hat.requires_grad_(True)
            y_pred = forward_op(x0hat.squeeze(1))
            loss_y = torch.mean((y_pred - observations) ** 2)  # MSE loss
            log_likelihood_grad = torch.autograd.grad(
                loss_y.sum(), x0hat, retain_graph=True
            )[0].detach()

        log_likelihood_grad = self.noise_sampler.apply_C(log_likelihood_grad)

        # print("x.shape:", x.shape, "y_inp.shape:", y_inp.shape, "log_likelihood_grad.shape:", log_likelihood_grad.shape, "grid.shape:", grid.shape, "t.shape:", t.shape)
        pred_h = (
            self.h_trans(
                x=x, x0hat=x0hat, y=y_inp, log_grad=log_likelihood_grad, pos=grid, t=t
            ).unsqueeze(1)
            / std_t
        )

        if self.cond_model_type == "raw":  # the network output is nabla log p
            pred_h = self.noise_sampler.apply_C(pred_h)
        elif (
            self.cond_model_type == "C_sqrt"
        ):  # the network output is C^{1/2} nabla log p
            pred_h = self.noise_sampler.apply_Csqrt(pred_h)
        elif self.cond_model_type == "C":  # the network output is C nabla log p
            pred_h = pred_h
        else:
            raise NotImplementedError

        pred = pred + pred_h

        return pred, None
