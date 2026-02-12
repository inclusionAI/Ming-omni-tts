import torch
from torch import nn
import torch.nn.functional as F


class Solver:
    def __init__(self, func, y0, sigma=0.25, temperature=1.5) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            dy = dt * f0
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1

            noise = torch.randn_like(y0) 
            shift = self.sigma * (self.temperature ** 0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift

        return solution

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        cond,
        target,
        latent_history,
        mask,
        patch_size,
    ):
        x1 = target
        batch, dtype = x1.shape[0], x1.dtype
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        x = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.model(x=x, t=time, c=cond, latent_history=latent_history, mask=mask.to(torch.bool))
        pred = pred[:, -patch_size:, :]

        loss = F.mse_loss(pred, flow, reduction="none")
        mask = (mask == 1)
        loss = loss[mask]

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        noise,
        c,
        latent_history,
        steps=10,
        cfg_scale=1.0,
        sway_sampling_coef=-1.0,
        use_epss=True,
        patch_size=1,
        sigma=0.25,
        temperature=1.5
    ):
        def fn(t, x):
            if cfg_scale < 1e-5:
                pred = self.model(
                    x=x,
                    time=t,
                    latent_history=latent_history
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                cfg_scale=cfg_scale,
                patch_size=patch_size,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_scale

        y0 = noise.transpose(1, 2)
        t_start = 0

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=noise.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=noise.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t)
        sampled = trajectory[-1]
        out = sampled

        return out, trajectory
