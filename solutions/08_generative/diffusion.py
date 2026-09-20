"""Generative history III: diffusion (DDPM 2020 -> DDIM 2020 -> guidance).

History: score matching (2005-2011, Hyvarinen/Song) + DDPM's epsilon
reparameterization made likelihood-grade image models; DDIM showed the
Markov chain was optional (deterministic, strided sampling); classifier
guidance (2021) then classifier-FREE guidance traded a conditional and an
unconditional prediction for prompt adherence. Cosine schedule (2021)
fixed DDPM's linear schedule destroying information too fast at low res.
"""

import math

import torch


def linear_beta_schedule(T: int, b0: float = 1e-4, b1: float = 0.02) -> torch.Tensor:
    return torch.linspace(b0, b1, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule, clipped to [1e-5, 0.999]."""
    steps = torch.arange(T + 1).float()
    ab = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = (1.0 - ab[1:] / ab[:-1]).clamp(1e-5, 0.999)
    return betas


def classifier_free_guidance(eps_cond: torch.Tensor, eps_uncond: torch.Tensor,
                             w: float = 7.5) -> torch.Tensor:
    """eps_uncond + w * (eps_cond - eps_uncond); w=0 -> uncond, w=1 -> cond."""
    return eps_uncond + w * (eps_cond - eps_uncond)


class DiffusionSchedule:
    def __init__(self, T: int, kind: str = "linear"):
        self.T = T
        betas = linear_beta_schedule(T) if kind == "linear" else cosine_beta_schedule(T)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_1mab = (1.0 - self.alpha_bar).sqrt()

    def _coef(self, table: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        return table[t].view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._coef(self.sqrt_ab, t, x0.shape) * x0 + \
            self._coef(self.sqrt_1mab, t, x0.shape) * noise

    def ddpm_step(self, x_t: torch.Tensor, t_idx: int, eps_pred: torch.Tensor,
                  z: torch.Tensor | None = None) -> torch.Tensor:
        """One DDPM reverse step with epsilon parameterization."""
        b = self.betas[t_idx]
        ab, ab_prev = self.alpha_bar[t_idx], self.alpha_bar[t_idx - 1] if t_idx > 0 else 1.0
        x0_hat = (x_t - self.sqrt_1mab[t_idx] * eps_pred) / self.sqrt_ab[t_idx]
        coef_x0 = self.sqrt_ab[t_idx - 1] * b / (1 - ab) if t_idx > 0 else 1.0
        coef_xt = self.alphas[t_idx].sqrt() * (1 - ab_prev) / (1 - ab)
        mean = coef_x0 * x0_hat + coef_xt * x_t
        if t_idx == 0:
            return mean
        sigma = (((1 - ab_prev) / (1 - ab)) * b).sqrt()
        return mean + sigma * (torch.randn_like(x_t) if z is None else z)

    def ddim_step(self, x_t: torch.Tensor, t_idx: int, prev_idx: int,
                  eps_pred: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """Deterministic (eta=0) strided reverse step."""
        ab, ab_prev = self.alpha_bar[t_idx], self.alpha_bar[prev_idx]
        x0_hat = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        direction = (1 - ab_prev - sigma ** 2).sqrt() * eps_pred
        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
        return ab_prev.sqrt() * x0_hat + direction + noise
