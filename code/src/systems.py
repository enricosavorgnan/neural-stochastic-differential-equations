"""
Implementation of several types of SDE compatible with the LatentSDE module.
"""
import torch
import torchsde

from typing import Sequence

from fsspec.asyn import private


class StochasticLorenz:
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """

    def __init__(self, sde_type : str = "ito", noise_type : str = "diagonal", a: Sequence = (10., 28., 8 / 3), b: Sequence = (.1, .28, .3)):
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.a = a
        self.b = b

    def f(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=[1, 1, 1], dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=[1, 1, 1], dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, method, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        X = torchsde.sdeint(self, x0, ts, method = method)
        if normalize:
            mean, std = torch.mean(X, dim=(0, 1)), torch.std(X, dim=(0, 1))
            X.sub_(mean).div_(std).add_(torch.randn_like(X) * noise_std)
        return X



class ClimateModel:
    """
    Climate Model SDE, as in Benzi, Sutera, Vulpiani, Parisi, A Theory of Stochastic Resonance in Climatic Change, 1983:

        dX = (aX - X^3 + b*cos(omega*t))dt + epsilon*dW

    Parameters values from the same paper
    """

    def __init__(self, sde_type : str = "ito", noise_type : str = "diagonal", a : float = 1, b : float = 0.1, omega : float = 0.1, epsilon : float = 0.53):
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.a = a
        self.b = b
        self.omega = omega
        self.eps = epsilon

    def f(self, t, x):
        drift = self.a*x - x**3 + self.b*torch.cos(self.omega*t)
        return drift

    def g(self, t, x):
        # map self.eps to the same shape as x
        self.eps *= torch.ones_like(x)
        return self.eps

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, method, normalize):
        # map x0, ts from points to
        X = torchsde.sdeint(self, x0, ts, method=method)
        if normalize:
            mean, std = torch.mean(X), torch.std(X)
            X.sub_(mean).div_(std).add_(torch.randn_like(X) * noise_std)
        print(f"samples done")
        return X



class SIRModel:
    """
    Stochastic SIR Model as in Tornatore, Buccellato, Vetro, Stability of a stochastic SIR system, 2004

    dS = ( mu - mu S(t) - beta(t) I(t)S(t) ) dt - sigma S(t)I(t)dW(t)
    dI = ( beta(t)I(t)S(t) - gamma I(t) - mu(t) ) dt + sigma S(t)I(t)dW(t)

    Time unit: 2 week
    The choice is motivated by the fact that choosing the standard time unit equal 1 day would have lead the system to
    have too tiny parameters to reconstruct.

    NB: The model, to be meaningful, should respect different constraints:
    - S(t) >= 0, I(t) >= 0 for all t
    - S(t) + I(t) <= 1 for all t
    - Initial conditions should respect the same constraints.

    In order to enforce these constraints, we can either:
    - Map variables using log-ratios, e.g. S' = log(S/(1-S-I)), I' = log(I/(1-S-I))
    - Do a simple projection
    Since the first option would introduces second derivatives (hessian matrices) in the drift and diffusion,
    we opt for the second choice, which is ruder but more efficient.
    We keep stochastic fluctuations small enough to avoid the system divergying from the meaningful region.
    """

    def __init__(self, sde_type : str = "ito",
                 noise_type : str = "diagonal",
                 mu : float | int = 1/80/26,
                 R0 : float | int = 2.5,
                 gamma : float | int = 1,
                 sigma : float | int = 0.2):
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.mu = mu
        self.R0 = R0
        self.gamma = gamma
        self.sigma = sigma
        self.beta = self.R0 * (self.gamma + self.mu)

    def _project_simplex(self, x):
        """
        Vectorized projection onto the valid epidemiological simplex domain:
        S >= 0, I >= 0, and S + I <= 1.
        """
        x_safe = torch.clamp(x, min=1e-5, max=1.0)
        S, I = torch.split(x_safe, split_size_or_sections=[1, 1], dim=-1)

        total = S + I
        scale = torch.where(total > 1.0, 1.0 / total, torch.ones_like(total))

        return S * scale, I * scale

    def f(self, t, x):
        S, I = self._project_simplex(x)

        dS = self.mu - self.mu*S - self.beta*I*S
        dI = self.beta*I*S - self.gamma*I - self.mu*I
        return torch.cat([dS, dI], dim=1)

    def g(self, t, x):
        S, I = self._project_simplex(x)

        dS = -self.sigma*S*I
        dI = self.sigma*S*I
        return torch.cat([dS, dI], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, method, normalize):
        s0, i0 = self._project_simplex(x0)
        x0_safe = torch.cat([s0, i0], dim=-1)

        X = torchsde.sdeint(self, x0_safe, ts, method=method)

        # Clamped projection to the generated trajectory tensor
        S, I = self._project_simplex(X)
        X = torch.cat([S, I], dim=-1)

        if normalize:
            mean, std = torch.mean(X, dim=(0, 1)), torch.std(X, dim=(0, 1))
            X.sub_(mean).div_(std).add_(torch.randn_like(X) * noise_std)
        print(f"samples done")
        return X
