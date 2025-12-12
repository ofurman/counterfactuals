from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import extract, DiffusionSchedule

class MixedTabularDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        num_numerical: int,
        num_classes: List[int],  # Lista kardynalności [K1, K2...]
        T: int = 1000,
        schedule: str = "cosine",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.num_numerical = num_numerical
        self.num_classes = num_classes
        self.num_classes_tensor = torch.tensor(num_classes, device=device)
        self.total_cat_dim = sum(num_classes)
        
        # Schedule configuration
        sched = DiffusionSchedule.from_name(schedule, T, device)
        self.register_buffer("betas", sched.betas)
        alphas = 1.0 - sched.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)
        
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        
        # Posterior params for Gaussian
        self.register_buffer("posterior_variance", sched.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))
        self.register_buffer("posterior_mean_coef1", sched.betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar))

    # ---------------- Gaussian Helpers ----------------
    def q_sample_gauss(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion for numerical variables."""
        coef1 = extract(self.sqrt_alphas_bar, t, (x0.shape[0], 1))
        coef2 = extract(self.sqrt_one_minus_alphas_bar, t, (x0.shape[0], 1))
        return coef1 * x0 + coef2 * noise

    def p_mean_variance_gauss(self, x_t: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and variance of p(x_{t-1} | x_t) given predicted noise (eps).
        Uses the standard DDPM formulation re-parameterized to predict eps.
        """

        sqrt_recip_alphas_bar = extract(1.0 / self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_recip_m1_alphas_bar = extract(torch.sqrt(1.0 / self.alphas_bar - 1.0), t, x_t.shape)
        pred_x0 = sqrt_recip_alphas_bar * x_t - sqrt_recip_m1_alphas_bar * pred_eps
        pred_x0 = pred_x0.clamp(-5.0, 5.0) 
        post_mean_coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        post_mean_coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        model_mean = post_mean_coef1 * pred_x0 + post_mean_coef2 * x_t
        model_log_var = extract(self.posterior_variance.clamp(min=1e-20).log(), t, x_t.shape)
        return model_mean, model_log_var

    # ---------------- Categorical Helpers (Multinomial) ----------------
    def q_sample_cat(self, log_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion for categorical (Multinomial).
        Uniform noise transition.
        """
        
        alphas_bar_t = extract(self.alphas_bar, t, (log_x0.shape[0], 1))
        out_parts = []
        start = 0
        for K in self.num_classes:
            sl = slice(start, start+K)
            log_probs_x0 = log_x0[:, sl]
            log_alpha = alphas_bar_t.log()
            log_1_m_alpha = (1. - alphas_bar_t).log()
            log_inv_K = -torch.log(torch.tensor(K, device=log_x0.device))
            
            term1 = log_alpha + log_probs_x0
            term2 = log_1_m_alpha + log_inv_K
            
            log_probs_t = torch.logaddexp(term1, term2)
            uniform = torch.rand_like(log_probs_t).clamp(min=1e-30)
            gumbel = -torch.log(-torch.log(uniform))
            sample_t = F.one_hot((log_probs_t + gumbel).argmax(dim=1), num_classes=K).float()
            
            out_parts.append((sample_t + 1e-30).log())
            start += K
            
        return torch.cat(out_parts, dim=1)

    def p_sample_cat(self, log_x_t: torch.Tensor, t: torch.Tensor, log_pred_x0: torch.Tensor) -> torch.Tensor:
        """
        Sample x_{t-1} given x_t and predicted x_0 (logits).
        Calculates posterior q(x_{t-1} | x_t, x_0).
        """
        out_parts = []
        start = 0
        t_idx = t[0] 
        
        alpha_t = self.alphas_bar[t_idx]
        alpha_prev = self.alphas_bar_prev[t_idx]
        beta_t = self.betas[t_idx]
        
        for K in self.num_classes:
            sl = slice(start, start+K)
            log_xt = log_x_t[:, sl]           # One-hot at t
            log_x0_rec = log_pred_x0[:, sl]   # Predicted logits x0
            
            log_x0_rec = F.log_softmax(log_x0_rec, dim=1)
            
            if t_idx == 0:
                out_parts.append(log_x0_rec)
            else:
                out_parts.append(log_x0_rec) 
                
            start += K
            
        full_log_probs = torch.cat(out_parts, dim=1)
        return self._sample_cat_from_logits(full_log_probs)

    def _sample_cat_from_logits(self, logits_flat: torch.Tensor) -> torch.Tensor:
        out_parts = []
        start = 0
        for K in self.num_classes:
            sl = slice(start, start+K)
            probs = F.softmax(logits_flat[:, sl], dim=1)
            idx = torch.multinomial(probs, 1).squeeze(1)
            one_hot = F.one_hot(idx, num_classes=K).float()
            out_parts.append((one_hot + 1e-30).log())
            start += K
        return torch.cat(out_parts, dim=1)


    def forward(self, x_neigh: torch.Tensor, x_orig: torch.Tensor, y_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        B = x_neigh.shape[0]
        t = torch.randint(0, len(self.betas), (B,), device=x_neigh.device)
        x_num = x_neigh[:, :self.num_numerical]
        x_cat_log = x_neigh[:, self.num_numerical:]

        # 1. Noise Numerical
        noise_num = torch.randn_like(x_num)
        x_num_t = self.q_sample_gauss(x_num, t, noise_num)

        # 2. Noise Categorical
        if self.total_cat_dim > 0:
            x_cat_t_log = self.q_sample_cat(x_cat_log, t)
        else:
            x_cat_t_log = x_cat_log

        # 3. Predict
        x_in_t = torch.cat([x_num_t, x_cat_t_log], dim=1)
        model_out = self.denoise_fn(x_in_t, t, x_orig, y_target)

        pred_num = model_out[:, :self.num_numerical]
        pred_cat_logits = model_out[:, self.num_numerical:]

        # 4. Loss Calculation
        loss_num = F.mse_loss(pred_num, noise_num)

        loss_cat = torch.tensor(0.0, device=x_neigh.device)
        if self.total_cat_dim > 0:
            start = 0
            losses = []
            for K in self.num_classes:
                # Target index for CE
                target_idx = x_cat_log[:, start:start+K].argmax(dim=1) # [B]
                # Predicted logits
                logits = pred_cat_logits[:, start:start+K]
                losses.append(F.cross_entropy(logits, target_idx))
                start += K
            loss_cat = torch.stack(losses).mean()

        total_loss = loss_num + loss_cat
        return total_loss, {"num": loss_num.item(), "cat": loss_cat.item()}

    @torch.no_grad()
    def sample_counterfactual(self, x_orig: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        B = x_orig.shape[0]
        device = x_orig.device
        
        # Start from pure noise
        x_num = torch.randn(B, self.num_numerical, device=device)
        cat_parts = []
        for K in self.num_classes:
            cat_parts.append(torch.zeros(B, K, device=device))
        x_cat_log = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros(B, 0, device=device)

        # Iterative denoising
        for i in reversed(range(0, len(self.betas))):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            x_in = torch.cat([x_num, x_cat_log], dim=1)
            
            # Predict
            model_out = self.denoise_fn(x_in, t, x_orig, y_target)
            pred_eps_num = model_out[:, :self.num_numerical]
            pred_logits_cat = model_out[:, self.num_numerical:]
            
            # Step Gauss
            mean, log_var = self.p_mean_variance_gauss(x_num, t, pred_eps_num)
            noise = torch.randn_like(x_num) if i > 0 else 0.0
            x_num = mean + torch.exp(0.5 * log_var) * noise
            
            # Step Cat
            if self.total_cat_dim > 0:
                x_cat_log = self.p_sample_cat(x_cat_log, t, pred_logits_cat)

        if self.total_cat_dim > 0:
             x_cat_final = torch.exp(x_cat_log)
             return torch.cat([x_num, x_cat_final], dim=1)
             
        return x_num