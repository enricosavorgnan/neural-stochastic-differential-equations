import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsde

import os
import yaml
import datetime

from typing import Callable, Sequence, Optional
import tqdm

import utils


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x



class Decoder(nn.Module):

    def __init__(self, input_size : int, hidden_size : int, output_size : int):
        super().__init__()

        self.linear = nn.Linear(input_size, hidden_size)
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.linear(x)
        x = self.softplus(x)
        x = self.linear2(x)
        return x



class Drift(nn.Module):

    def __init__(self, layer_numbers : int, layer_sizes : list[int]):
        super().__init__()

        # Layers: sequence of linear + softplus for n-1 layers, and finally a linear layer for the last layer
        self.layers = []
        for i in range(layer_numbers - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.Softplus())
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))



class Diffusion(nn.Module):

    def __init__(self, layer_numbers : int, layer_sizes : list[list[int]]):
        super().__init__()

        # Element-wise function for the diffusion, so to satisfy diagonal noise (required for efficient computation)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i][0], layer_sizes[i][1]),
                nn.Softplus(),
                nn.Linear(layer_sizes[i][1], layer_sizes[i][2]),
                nn.Sigmoid()
            )
            for i in range(layer_numbers - 1)
        ])

    # define an iterable to iterate through the layers
    def __iter__(self):
        return iter(self.layers)



class LatentSDE(nn.Module):

    def __init__(self, input_size, latent_size, context_size, hidden_size):
        super().__init__()

        # Core structure
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, output_size=context_size)

        self.drift_posterior = Drift(layer_numbers=3, layer_sizes=[latent_size+context_size, hidden_size, hidden_size, latent_size])
        self.drift_prior = Drift(layer_numbers=3, layer_sizes=[latent_size, hidden_size, hidden_size, latent_size])
        self.diffusion = Diffusion(layer_numbers = latent_size, layer_sizes=[[1, hidden_size, 1] for _ in range(latent_size)])

        self.decoder = Decoder(input_size=latent_size, hidden_size=hidden_size, output_size=input_size)

        # Context,
        # Will store the encoded information regard the future trajectory at each timestep
        self._context = None

        # Probabilities
        self.qz0 = nn.Linear(context_size, latent_size + latent_size)   # prior gaussian distribution
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))       # posterior learned mean
        self.pz0_log_std = nn.Parameter(torch.zeros(1, latent_size))    # posterior learned log-standard-deviation


    def set_context(self, context):
        self._context = context


    # --------------------------
    # Utilities for the adjoints
    # f = Drift Posterior
    # h = Drift Prior
    # g = Diffusion (Both Prior and Posterior)

    def f(self, t, x):
        ts, context = self._context
        idx = torch.min(torch.searchsorted(ts, t, right=True), len(ts)-1)
        #Interpolation of the context vector at time t
        if idx == 0:
            # t is exactly at or before the first timestamp
            interp_ctx = context[0]
        elif idx == len(ts):
            # t is exactly at or after the last timestamp
            interp_ctx = context[-1]
        else:
            # Extract bounding timestamps
            t0 = ts[idx - 1]
            t1 = ts[idx]

            # Extract bounding context vectors
            ctx0 = context[idx - 1]
            ctx1 = context[idx]

            # Calculate the interpolation weight (alpha in [0, 1])
            alpha = (t - t0) / (t1 - t0)

            # Blend the context vectors
            interp_ctx = ctx0 + alpha * (ctx1 - ctx0)
        return self.drift_posterior(torch.cat((x, interp_ctx), dim=1))

    def h(self, t, x):
        return self.drift_prior(x)

    def g(self, t, x):
        x = torch.split(x, split_size_or_sections=1, dim = 1)
        x = [diff(x_i) for (diff, x_i) in zip(self.diffusion, x)]
        return torch.cat(x, dim=1)


    def forward(self, X, ts, noise_std, adjoint : bool = False, method : str = 'euler', dt : float = 0.01):

        # Get context from moving BACKWARD in time, i.e., from the last X to the first.
        # This helps the SDE by encoding the future information into a single point
        context = self.encoder(torch.flip(X, dims=(0,)))
        context = torch.flip(context, dims=(0,))
        self.set_context((ts,context))

        # Recover prior distribution
        qz0_mean, qz0_log_std = self.qz0(context[0]).chunk(chunks=2, dim=1)
        # Reparameterization trick
        z0 = qz0_mean + qz0_log_std.exp() * torch.randn_like(qz0_mean)

        # Reconstruct SDE
        if adjoint:
            params = (
                    (context, ) +
                    tuple(self.drift_posterior.parameters()) +
                    tuple(self.drift_prior.parameters()) +
                    tuple(self.diffusion.parameters())
                )
            Z, log_ratio = torchsde.sdeint_adjoint(sde=self, y0=z0, ts=ts, adjoint_params=params, dt=dt, logqp=True, method=method)
        else:
            Z, log_ratio = torchsde.sdeint(sde=self, y0=z0, ts=ts, dt=dt, logqp=True, method=method)

        # Decode starting from the solved SDE
        X_new = self.decoder(Z)

        # Compute KL Divergence between the prior and posterior distributions at time 0
        X_distribution = torch.distributions.normal.Normal(loc=X_new, scale=noise_std)
        log_p_X = X_distribution.log_prob(X).sum(dim=(0,2)).mean(dim=0)

        qz0 = torch.distributions.normal.Normal(loc=qz0_mean, scale=qz0_log_std.exp())
        pz0 = torch.distributions.normal.Normal(loc=self.pz0_mean, scale=self.pz0_log_std.exp())

        kl = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        path = log_ratio.sum(dim=0).mean(dim=0)
        return log_p_X, kl + path


    def sample(self, batch_size, ts, brownian_motion=None, **kwargs):
        epsilon = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_log_std.exp() * epsilon

        Z = torchsde.sdeint(sde=self, y0 = z0, ts=ts, bm=brownian_motion, names={'drift': 'h'}, dt=kwargs.get('dt', 1e-2))
        X = self.decoder(Z)
        return X



class LatentSDETrainer:
    """
    Class to train the Latent SDE model.
    It takes care of loading the config, generating the data, defining the model, and training it.
    """

    def __init__(self, config_path : str):
        # Open Config file
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file: \n{config_path}\n not found. \nExiting.")
            exit(1)

        self.sde_name = config['sde']['name']
        self.sde_system = config['sde']['system']
        self.sde_type = config['sde']['sde_type']
        self.t_span = config['sde']['t_span']
        self.noise_std = config['sde']['noise_std']
        self.method = config['sde']['method']
        self.adjoint = config['sde']['adjoint']
        self.levy_area_type = config['sde']['levy_area']
        self.dt = config['sde']['dt']

        self.data_size = config['model']['data_size']
        self.batch_size = config['model']['batch_size']
        self.latent_size = config['model']['latent_size']
        self.context_size = config['model']['context_size']
        self.hidden_size = config['model']['hidden_size']

        self.n_iters = config['training']['n_iters']
        self.device = config['training']['device']
        self.optimizer_name = config['training']['optimizer']
        self.lr_scheduler_name = config['training']['lr_scheduler']
        self.lr_init = config['training']['lr_init']
        self.lr_gamma = config['training']['lr_gamma']
        self.lr_step_size = config['training']['lr_step_size']
        self.kl_scheduler_name = config['training']['kl_scheduler']
        self.kl_annealed_iters = config['training']['kl_iters']
        self.pause_every = config['training']['pause']

        self.save_model = config['save']['model']
        self.save_model_every = config['save']['model_every']
        self.save_plot = config['save']['plot']
        self.save_data = config['save']['data']
        self.n_samples = config['save']['n_samples']
        self.plot_dim = config['save']['plot_dim']

        self.data_path = config['path']['data']
        self.model_path = config['path']['model']
        self.plot_path = config['path']['plot']

        self.data = config.get('data', None)

        # Generate Data
        if not self.data:
            self.generate_data()

        # Define Latent SDE to optimize
        self.latent_sde = LatentSDE(
                input_size = self.data_size,
                latent_size = self.latent_size,
                context_size = self.context_size,
                hidden_size = self.hidden_size
            ).to(self.device)

        # Instantiate Optimizer and Scheduler for Learning Rate and KL Divergence
        self.lr_scheduler = None
        self.optimizer = None
        self.kl_scheduler = None
        self.configure_optimizer_scheduler()


    def generate_data(self):
        """
        Generates synthetic data according to the specified SDE type and stores it in self.data.
        """
        _X0 = torch.randn(self.batch_size, self.data_size, device=self.device)
        ts = torch.linspace(self.t_span[0], self.t_span[1], steps=100, device=self.device)
        X = eval(self.sde_system).sample(_X0, ts, self.noise_std, normalize=True)
        return X, ts


    def configure_optimizer_scheduler(self):
        """
        Configures the optimizer and the scheduler with the stored information.
        Allowed optimizers: `SGD`, `adam`
        Allowed schedulers: `constant`, `step`, `exponential`
        """
        assert self.optimizer_name.lower() in ['sgd', 'adam'], f"Optimizer {self.optimizer_name} not supported. Allowed optimizers: `SGD`, `adam`"
        assert self.lr_scheduler_name.lower() in ['constant', 'step', 'exponential'], f"Scheduler {self.lr_scheduler_name} not supported. Allowed schedulers: `constant`, `step`, `exponential`"
        assert self.kl_scheduler_name.lower() in ['constant', 'linear'], f"KL Scheduler {self.kl_scheduler_name} not supported. Allowed KL schedulers: `constant`, `linear`"

        optimizer_class = torch.optim.Adam if self.optimizer_name == 'adam' else torch.optim.SGD
        self.optimizer = optimizer_class(self.latent_sde.parameters(), lr=self.lr_init)

        if self.lr_scheduler_name.lower() == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma = self.lr_gamma, step_size = self.lr_step_size)
        elif self.lr_scheduler_name.lower() == 'exponential':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = self.lr_gamma)
        else:
            # is constant, so we can use a lambda scheduler that always returns 1
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)

        if self.kl_scheduler_name.lower() == 'linear':
            self.kl_scheduler = lambda step: min(1.0, step / self.kl_annealed_iters)
        else:
            # is constant, so we can use a lambda scheduler that always returns 1
            self.kl_scheduler = lambda step: 1.0


    def train(self, data):
        """
        Train method
        """
        # Setting time for future savings
        now = datetime.datetime.now().strftime("%m-%d_%H-%M")

        # Retrieve data
        X, ts = self.data

        # Sample a brownian motion
        # Just for visualization
        brownian_motion = torchsde.BrownianInterval(
            t0 = self.t_span[0],
            t1 = self.t_span[1],
            size = (self.batch_size, self.latent_size,),
            device = self.device,
            levy_area_approximation = self.levy_area_type
        )

        # Start training
        for iteration in tqdm.tqdm(range(1, self.n_iters+1)):
            self.latent_sde.zero_grad()
            log_p_X, kl = self.latent_sde(X, ts, noise_std = self.noise_std, adjoint = self.adjoint, method = self.method, dt = self.dt)

            loss = - log_p_X + kl * self.kl_scheduler
            if torch.isnan(loss):
                break
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.kl_scheduler.step()

            if iteration % self.pause_every == 0:
                print(f"Iter: {iteration}/{self.n_iters}, \tLoss: {loss.item():.4f}, \tLog p(X): {log_p_X.item():.4f}, \tKL: {kl.item():.4f}")
                # store checkpoint
                self.model_saver(time=now, checkpoint=True, iteration=iteration)

        if self.save_data:
            self.data_saver(time=now)
        if self.save_model:
            self.model_saver(time=now)
        if self.save_plot:
            self.plot_saver(bm=brownian_motion, time=now, data=X, ts=ts)


    def plot_saver(self, bm, time, data, ts):
        """
        Plots the trajectories of the data and the samples from the trained SDE, and saves the plot to the specified path.
        """
        samples = self.latent_sde.sample(batch_size=data.size(1), ts=ts, bm=bm).cpu().numpy()

        if self.plot_dim == 1:
            utils.plot_1d_latent_sde(ts=ts, X_data=data, X_samples=samples, time=time, plot_path = self.plot_path, name = self.sde_name)
        elif self.plot_dim == 2:
            utils.plot_2d_latent_sde(ts=ts, X_data=data, X_samples=samples, time=time, plot_path = self.plot_path, name = self.sde_name)
        else:
            utils.plot_3d_latent_sde(ts=ts, X_data=data, X_samples=samples, time=time, plot_path = self.plot_path, name = self.sde_name)


    def model_saver(self, time, checkpoint : bool = False, iteration=None):
        """
        Saves the model to the specified path.
        The model is saved with the name: sde_{sde_name}_time_{time}.pt
        """
        os.makedirs(self.model_path, exist_ok=True)
        if checkpoint:
            model_save_path = os.path.join(self.model_path, f"sde_{self.sde_name}_time_{time}_iter_{iteration}/{self.n_iters}.pt")
        else:
            model_save_path = os.path.join(self.model_path, f"sde_{self.sde_name}_time_{time}.pt")
        torch.save(self.latent_sde.state_dict(), model_save_path)
        print(f"Model saved at: {model_save_path}")


    def data_saver(self, time):
        """
        Saves the data to the specified path.
        """
        os.makedirs(self.data_path, exist_ok=True)
        data_save_path = os.path.join(self.data_path, f"data_{self.sde_name}_time_{time}.pt")
        torch.save(self.data, data_save_path)
        print(f"Data saved at: {data_save_path}")



if __name__ == "__main__":
    sde = LatentSDE()
    sde.train()