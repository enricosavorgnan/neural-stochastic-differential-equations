"""
Implementation of several types of SDE compatible with the LatentSDE module.
"""
import torch
import torchsde

from typing import Sequence


class StochasticLorenz:
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.1, .28, .3)):
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
    Climate Model SDE, following Benzi, Sutera, Vulpiani, Parisi, A Theory of Stochastic Resonance in Climatic Change, 1983:

        dX = (aX - X^3 + b*cos(omega*t))dt + epsilon*dW

    Parameters values from the same paper
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a : float = 1, b : float = 0.1, omega : float = 0.01, epsilon : float = 0.35):
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
        return X