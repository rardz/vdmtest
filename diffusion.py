import math
import torch
from torch.nn import functional as F

from networks import eta_to_gamma, get_eta_scale

def unsqueeze_x_as_y(x, y):
    if x.ndim == y.ndim:
        return x

    assert x.size(0) == y.size(0)
    return x.view(-1, *([1, ] * (y.ndim-1)) )


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps=1000,
        match_obj='eps',  # eps, x_start
        **unused,
    ):
        self.num_timesteps = num_timesteps
        assert match_obj in ['eps', 'x_start']
        self.match_obj = match_obj

    def get_alpha_square(self, gamma_t):
        """
        snr_t = exp(-gamma_t)
        (alpha_t)^2 = snr_t / (snr_t +1 ) = 1 / (exp(gamma_t) + 1)
        """
        return torch.sigmoid(-gamma_t)

    def get_alpha(self, gamma_t):
        return torch.sqrt(self.get_alpha_square(gamma_t))

    def get_variance(self, gamma_t):
        return torch.sigmoid(gamma_t)

    def get_log_variance(self, gamma_t):
        return  F.softplus(gamma_t, beta=-1)

    def get_sigma(self, gamma_t):
        return torch.sqrt(self.get_variance(gamma_t))

    def get_variance_t_on_s(self, gamma_t, gamma_s):
        """
        \sigma^2_{t|s}
        """
        return - torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

    def get_frac_variance_t_on_s(self, gamma_t, gamma_s):
        """
        \sigma^2_{t|s} / (1 - \sigma^2_{t|s})
        """
        return torch.expm1(F.softplus(gamma_t) - F.softplus(gamma_s))

    def get_variance_s_div_t(self, gamma_t, gamma_s):
        """
        (simga_s / sigma_t)^2 = (1 + exp(-gamma_t)) / (1 + exp(-gamma_s)) 
                              = (1 + exp(-gamma_t)) * simgoid(gamma_s)
        """
        return (1 + torch.exp(-gamma_t)) * torch.sigmoid(gamma_s)


    def q_sample(self, x_start, gamma_t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        else:
            assert noise.shape == x_start.shape

        gamma_t = unsqueeze_x_as_y(gamma_t, x_start)
        alpha_t, sigma_t = self.get_alpha(gamma_t), self.get_sigma(gamma_t)

        return alpha_t * x_start + sigma_t * noise


    def predict_xstart_from_eps(self, z_t, gamma_t, eps):
        '''
        z_t = alpha_t * x_start + simga_t * eps
        x_start = z_t / alpha_t  - simga_t / alpha_t * eps

        has:
            (1 / alpha_t)^2 = 1 + exp(gamma_t)
            (simga_t / alpha_t)^2 = SNR(t)^-1 = exp(gamma_t)  
        '''
        assert z_t.shape == eps.shape

        gamma_t = unsqueeze_x_as_y(gamma_t, z_t)
        return torch.sqrt(torch.exp(gamma_t) + 1) * z_t - torch.exp(gamma_t / 2) * eps

    def predict_eps_from_xstart(self, z_t, gamma_t, pred_xstart):
        '''
        z_t = alpha_t * x_start + simga_t * eps
        eps = z_t / sigma_t - alpha_t / sigma_t * x_start

        has:
            1 / variance_t = 1 + exp(-gamma_t), so: 1 / sigma_t = sqrt(1 + exp(-gamma_t))
            alpha_t / sigma_t = exp(-gamma_t/2)
        '''
        assert z_t.shape == pred_xstart.shape

        gamma_t = unsqueeze_x_as_y(gamma_t, z_t)
        return torch.sqrt(1 + torch.exp(-gamma_t)) * z_t - torch.exp(-gamma_t/2) * pred_xstart




    def q_posterior(self, x_start, z_t, gamma_t, gamma_s):
        """
        Compute the mean and variance of the diffusion posterior:

        q(x_{t-1} | x_t, x_0)
        (alpha_t / alpha_s)^2 = (exp(gamma_s) + 1) / (exp(gamma_t))+1)
                              = (exp(gamma_s) + 1) * sigmoid(-gamma_t)
        """
        gamma_t = unsqueeze_x_as_y(gamma_t, x_start)
        gamma_s = unsqueeze_x_as_y(gamma_s, x_start)

        variance_t_on_s = self.get_variance_t_on_s(gamma_t, gamma_s)
        variance_s_div_t = self.get_variance_s_div_t(gamma_t, gamma_s)
        variance = variance_t_on_s * variance_s_div_t

        alpha_t_on_s = torch.sqrt(torch.exp(gamma_s) + 1) * torch.sqrt(torch.sigmoid(-gamma_t))
        recip_variance_t = 1 + torch.exp(-gamma_t)
        mean = alpha_t_on_s * variance_s_div_t * z_t + variance_t_on_s * self.get_alpha(gamma_s) * recip_variance_t * x_start

        return mean, variance


    def p_mean_variance(self,
                        model,
                        z_t,
                        model_kwargs,
                        gamma_t,
                        gamma_s,
                        clip_value=1,
                        denoised_fn=None):
        """
        Apply the model to get p(z_s | z_t)
        """

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_value:
                return x.clamp(-clip_value, clip_value)
            return x

        if self.match_obj == 'eps':
            pred_eps = model(z_t, **model_kwargs)
            pred_xstart = self.predict_xstart_from_eps(z_t, gamma_t, eps=pred_eps)
            pred_xstart = process_xstart(pred_xstart)
        else: # 'x_start':
            pred_xstart = model(z_t, **model_kwargs)
            pred_xstart = process_xstart(pred_xstart)
            pred_eps = self.predict_eps_from_xstart(z_t, gamma_t, pred_xstart=pred_xstart)

        model_mean, model_variance = self.q_posterior(x_start=pred_xstart,
                                                      z_t=z_t,
                                                      gamma_t=gamma_t,
                                                      gamma_s=gamma_s)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "pred_xstart": pred_xstart,
            "pred_eps": pred_eps,
        }


    def p_sample(self,
                 model,
                 z_t,
                 model_kwargs,
                 gamma_t,
                 gamma_s,
                 clip_value=1,
                 denoised_fn=None):

        out = self.p_mean_variance(
            model,
            z_t,
            model_kwargs,
            gamma_t,
            gamma_s,
            clip_value=clip_value,
            denoised_fn=denoised_fn
        )
        noise = torch.randn_like(z_t)
        sample = out["mean"] + torch.sqrt(out["variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    @torch.no_grad()
    def p_sample_loop(self,
                      model,
                      gammanet,
                      shape,
                      device,
                      clip_value=1,
                      noise_fn=torch.randn,
                      denoised_fn=None,
                      progress=False):
        img = noise_fn(shape, device=device)

        ti = torch.linspace(0, 1, self.num_timesteps+1, device=device).view(-1, 1)
        eta_t, eta_be, gamma_be = gammanet(ti)
        gamma, eta_norm = eta_to_gamma(eta_t, eta_be, gamma_be)
        indices = list(reversed(range(1, self.num_timesteps + 1)))
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = torch.full([shape[0]], fill_value=i, device=device, dtype=torch.int64)
            gamma_t = gamma[i].view(1).expand(shape[0])
            gamma_s = gamma[i-1].view(1).expand(shape[0])

            img = self.p_sample(
                model,
                img,
                {'time': t, 'eta_norm': eta_norm[i].view(1).expand(shape[0])},
                gamma_t,
                gamma_s,
                clip_value=clip_value,
                denoised_fn=denoised_fn,
            )['sample']

        return img


    def get_delta_gamma(self, eta_t, eta_s, eta_be, gamma_be):
        '''
        as:    gamma_t = gamma_0 + (gamma_1 - gamma_0) (eta_t - eta_0) / (eta_1 - eta_0)
        so:    gamma_t - gamma_s =  (eta_t - eta_s) (gamma_1 - gamma_0) / (eta_1 - eta_0)
        '''
        eta_scale = get_eta_scale(eta_be, gamma_be)
        delta = (eta_t - eta_s) * eta_scale
        return delta

    def weight_diff_pred_eps(self, eta_t, eta_s, eta_be, gamma_be):

        """
        Compute the weight for diff loss: (SNR(s) / SNR(t)) -1

        = exp(-gamma_s) / exp(-gamma_t) - 1
        = exp(gamma_t - gamma_s) - 1
        """
        delta = self.get_delta_gamma(eta_t, eta_s, eta_be, gamma_be)
        return torch.expm1(delta)

    def weight_diff_pred_xstart(self, eta_t, eta_s, eta_be, gamma_be, gamma_t):

        """
        Compute the weight for diff loss: (SNR(s) - SNR(t))

        = exp(-gamma_s) - exp(-gamma_t)
        = exp(-gamma_t) * (exp(gamma_t - gamma_s) - 1)
        """
        delta = self.get_delta_gamma(eta_t, eta_s, eta_be, gamma_be)
        return torch.expm1(delta) * torch.exp(-gamma_t)

    def prior_norm_kl(self, x_start, gamma_t):
        '''
        (-log(variance_t) + variance_t + (alpha_t^2 x ^2) - 1) / 2
        '''
        gamma_t = unsqueeze_x_as_y(gamma_t, x_start)
        return (- self.get_log_variance(gamma_t) + self.get_variance(gamma_t) + self.get_alpha_square(gamma_t) * x_start.pow(2) - 1) / 2

    def neg_log_pdf(self, x_start, gamma_t):
        gamma_t = unsqueeze_x_as_y(gamma_t, x_start)
        return ((1 - 2 / (1 + torch.sqrt(1 + gamma_t.exp()))) * x_start.pow(2) + F.softplus(gamma_t, beta=-1) + math.log(2 * math.pi)) / 2

    def training_losses(self, model, gammanet, x_start, t, noise=None, t_input=True):
        B, *_ = x_start.shape

        if noise is None:
            noise = torch.randn_like(x_start)

        ti = torch.cat([t, t-1]).float() / self.num_timesteps
        ti = ti.view(-1)

        eta_out, eta_be, gamma_be = gammanet(ti)
        eta_t, eta_s = eta_out.chunk(2, dim=0)
        gamma_t, eta_norm = eta_to_gamma(eta_t, eta_be, gamma_be)

        z_t = self.q_sample(x_start, gamma_t, noise)
        pred_out = model(z_t, time=t, eta_norm=eta_norm)
        if self.match_obj == 'eps':
            weight_diff = self.weight_diff_pred_eps(eta_t, eta_s, eta_be, gamma_be)
            loss_diff = (self.num_timesteps * 0.5) * (unsqueeze_x_as_y(weight_diff, x_start) * F.mse_loss(pred_out, noise, reduction='none')).mean()
        else:
            weight_diff = self.weight_diff_pred_xstart(eta_t, eta_s, eta_be, gamma_be, gamma_t)
            loss_diff = (self.num_timesteps * 0.5) * (unsqueeze_x_as_y(weight_diff, x_start) * F.mse_loss(pred_out, x_start, reduction='none')).mean()

        gamma_0, gamma_1 = gamma_be
        loss_prior = self.prior_norm_kl(x_start, gamma_1.view(1, 1).expand(B, -1)).mean()
        loss_rec = self.neg_log_pdf(x_start, gamma_0.view(1, 1).expand(B, -1)).mean()

        return {'diff': loss_diff, 'prior': loss_prior, 'rec': loss_rec}
