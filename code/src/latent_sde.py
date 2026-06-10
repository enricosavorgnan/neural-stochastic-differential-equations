"""
This file contains the implementation of the Latent SDE model and its training loop.
"""
import sys
import os
import yaml
import datetime
import tqdm
import argparse

import torch
import torch.nn as nn
import torchsde
from torch.utils.data import TensorDataset, DataLoader

import utils
from systems import StochasticLorenz, ClimateModel, SIRModel

# Increase the stack recursion limit to avoid issues
# with the adjoint method when using a large number of iterations or a complex model.
sys.setrecursionlimit(20000)


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
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
            nn.Softplus()) for i in range(layer_numbers-1)
        ])
        # append last layer without softplus
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



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
            for i in range(layer_numbers)
        ])


    def __iter__(self):
        return iter(self.layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class LatentSDE(nn.Module):

    def __init__(self, sde_type, noise_type, method, input_size, latent_size, context_size, hidden_size):
        super().__init__()
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.method = method

        # Core structure
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, output_size=context_size)

        self.drift_posterior = Drift(layer_numbers=3,
                                     layer_sizes=[latent_size+context_size, hidden_size, hidden_size, latent_size])
        self.drift_prior = Drift(layer_numbers=3,
                                 layer_sizes=[latent_size, hidden_size, hidden_size, latent_size])
        self.diffusion = Diffusion(layer_numbers = latent_size,
                                   layer_sizes=[[1, hidden_size, 1] for _ in range(latent_size)])

        self.decoder = Decoder(input_size=latent_size, hidden_size=hidden_size, output_size=input_size)

        # Context,
        # Will store the encoded information regard the future trajectory at each timestep
        self._context = None

        # Probabilities
        self.qz0 = nn.Linear(context_size, latent_size + latent_size)   # prior gaussian distribution
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))       # posterior learned mean
        self.pz0_log_std = nn.Parameter(torch.zeros(1, latent_size))    # posterior learned log-standard-deviation


    def set_context(self, context):
        """Simply set context variable"""
        self._context = context


    # --------------------------
    # Utilities for the adjoints
    # f = Drift Posterior
    # h = Drift Prior
    # g = Diffusion (Both Prior and Posterior)

    def f(self, t, x):
        ts, context = self._context
        idx = torch.min(torch.searchsorted(ts, t, right=True)).item()

        # Continuous methods require something stronger than just taking one extreme.
        # Here the solution is a classical interpolation.
        # N.B.: interpolation slows the process (around 0.5x)!
        if not self.method == 'euler':
            # Interpolation of the context vector at time t
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
        else:
            # If using Euler method, we can directly use the context at the closest timestamp without interpolation
            return self.drift_posterior(torch.cat((x, context[idx-1]), dim=1))

    def h(self, t, x):
        """Simply cross the layer"""
        return self.drift_prior(x)

    def g(self, t, x):
        """Cross the layer, but element-wise, ensuring diagonal noise"""
        x = torch.split(x, split_size_or_sections=1, dim = 1)
        x = [diff(x_i) for (diff, x_i) in zip(self.diffusion, x)]
        return torch.cat(x, dim=1) + 1e-4


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
            Z, pathwise_kl = torchsde.sdeint_adjoint(sde=self,
                                                   y0=z0,
                                                   ts=ts,
                                                   adjoint_params=params,
                                                   dt=dt,
                                                   logqp=True,
                                                   method=method)
        else:
            Z, pathwise_kl = torchsde.sdeint(sde=self, y0=z0, ts=ts, dt=dt, logqp=True, method=method)

        # Decode starting from the solved SDE
        X_new = self.decoder(Z)

        # Compute KL Divergence between the prior and posterior distributions at time 0
        X_distribution = torch.distributions.normal.Normal(loc=X_new, scale=noise_std)
        log_p_X = X_distribution.log_prob(X).sum(dim=(0,2)).mean(dim=0)

        qz0 = torch.distributions.normal.Normal(loc=qz0_mean, scale=qz0_log_std.exp())
        pz0 = torch.distributions.normal.Normal(loc=self.pz0_mean, scale=self.pz0_log_std.exp())

        distribution_kl = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        path_kl = pathwise_kl.sum(dim=0).mean(dim=0)
        return log_p_X, distribution_kl + path_kl


    def sample(self, batch_size, ts, brownian_motion=None, dt=0.01):
        epsilon = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_log_std.exp() * epsilon

        Z = torchsde.sdeint(sde=self,
                            y0 = z0,
                            ts=ts,
                            bm=brownian_motion,
                            names={'drift': 'h'},
                            dt=dt)
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
        self.dt = config['sde']['dt']
        self.noise_type = config['sde']['noise_type']
        self.noise_std = config['sde']['noise_std']
        self.method = config['sde']['method']
        self.adjoint = config['sde']['adjoint']
        self.levy_area_type = config['sde']['levy_area']
        self.sde_x0_type = config['sde']['x0_type']
        self.sde_x0_mean = config['sde']['x0_mean']
        self.sde_x0_std = config['sde']['x0_std']

        self.data_size = config['size']['data_size']
        self.dataset_size = config['size']['dataset_size']
        self.batch_size = config['size']['batch_size']
        self.latent_size = config['size']['latent_size']
        self.context_size = config['size']['context_size']
        self.hidden_size = config['size']['hidden_size']

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

        self.data_path = config['path']['data']
        self.model_path = config['path']['model']
        self.plot_path = config['path']['plot']

        self.data = config.get('data', None)
        self.model = config.get('model', None)
        self.checkpoint = int(config['checkpoint'])

        # Generate Data
        if not self.data:
            self.generate_data()
        else:
            self.load_data()

        # Define Latent SDE to optimize
        self.latent_sde = LatentSDE(
                sde_type = self.sde_type,
                noise_type= self.noise_type,
                method = self.method,
                input_size = self.data_size,
                latent_size = self.latent_size,
                context_size = self.context_size,
                hidden_size = self.hidden_size,
            ).to(self.device)
        if self.model:
            self.load_model()

        # Instantiate Optimizer and Scheduler for Learning Rate and KL Divergence
        self.lr_scheduler = None
        self.optimizer = None
        self.kl_scheduler = None
        self.configure_optimizer_scheduler()


    def load_model(self):
        """
        Loads the model from the specified path in self.model.
        The model is expected to be a state_dict of the LatentSDE class.
        """
        if not os.path.isfile(self.model):
            print(f"Model file: \n{self.model}\n not found. \nExiting.")
            exit(1)

        model_state_dict = torch.load(self.model, map_location=self.device)
        self.latent_sde.load_state_dict(model_state_dict)
        print(f"Model loaded from: {self.model}")


    def load_data(self):
        """
        Loads the data from the specified path in self.data.
        The data is expected to be a tuple of (X, ts) where X is the trajectory data and ts are the time steps.
        """
        if not os.path.isfile(self.data):
            print(f"Data file: \n{self.data}\n not found. \nExiting.")
            exit(1)

        self.data = torch.load(self.data, map_location=self.device)
        print(f"Data loaded from: {self.data}")


    def generate_data(self):
        """
        Generates synthetic data according to the specified SDE type and stores it in self.data.
        """
        if self.sde_x0_type == 'constant':
            _X0 = self.sde_x0_mean * torch.ones(self.dataset_size, self.data_size, device=self.device)
        elif self.sde_x0_type == 'random':
            _X0 = self.sde_x0_mean + torch.randn(self.dataset_size, self.data_size, device=self.device) * self.sde_x0_std
        ts = torch.linspace(self.t_span[0], self.t_span[1], steps=int((self.t_span[1] - self.t_span[0]) / self.dt), device=self.device)
        system_class = eval(self.sde_system)
        system = system_class(sde_type = self.sde_type)
        X = system.sample(x0=_X0, ts=ts, noise_std=self.noise_std, method=self.method, normalize=True)

        self.data = (X, ts)


    def generate_brownian_motion(self):
        """
        Generates a sample of Brownian motion with the same time steps and batch size as the data, and stores it in self.brownian_motion.
        This is used for visualization purposes.
        """
        brownian_motion = torchsde.BrownianInterval(
            t0 = self.t_span[0],
            t1 = self.t_span[1],
            size = (self.batch_size, self.latent_size,),
            device = self.device,
            levy_area_approximation = self.levy_area_type
        )
        return brownian_motion


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

        if self.checkpoint:
            self.lr_init = self.lr_init * (self.lr_gamma ** (self.checkpoint // self.lr_step_size))
            print(f"Checkpoint at iteration {self.checkpoint} detected. Adjusting initial learning rate to {self.lr_init:.6f} to account for previous training.")



    def plot_saver(self, bm, time, data, ts):
        """
        Plots the trajectories of the data and the samples from the trained SDE, and saves the plot to the specified path.
        """
        with torch.no_grad():
            samples = self.latent_sde.sample(batch_size=self.batch_size, ts=ts, brownian_motion=bm, dt=self.dt).detach().cpu().numpy()

        if self.data_size == 1:
            utils.plot_1d_latent_sde(ts=ts,
                                     X_data=data,
                                     X_samples=samples,
                                     time=time,
                                     plot_path = self.plot_path,
                                     name = self.sde_name,
                                     n_samples=self.n_samples)
        elif self.data_size == 2:
            utils.plot_2d_latent_sde(ts=ts,
                                     X_data=data,
                                     X_samples=samples,
                                     time=time,
                                     plot_path = self.plot_path,
                                     name = self.sde_name,
                                     n_samples=self.n_samples)
        else:
            utils.plot_3d_latent_sde(ts=ts,
                                     X_data=data,
                                     X_samples=samples,
                                     time=time,
                                     plot_path = self.plot_path,
                                     name = self.sde_name,
                                     n_samples=self.n_samples)


    def model_saver(self, time, checkpoint : bool = False, iteration=None):
        """
        Saves the model to the specified path.
        The model is saved with the name: sde_{sde_name}_time_{time}.pt
        """
        os.makedirs(self.model_path, exist_ok=True)
        if checkpoint:
            model_save_path = os.path.join(self.model_path, f"sde_{self.sde_name}_time_{time}_iter_{iteration}-{self.n_iters}.pt")
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


    def plot(self):
        """
        Plots the trajectories of the data and the samples from the trained SDE, and saves the plot to the specified path.
        """
        bm = self.generate_brownian_motion()

        self.plot_saver(bm=bm, time=datetime.datetime.now().strftime("%m-%d_%H-%M"), data=self.data[0], ts=self.data[1])


    def train(self):
        """
        Train method
        """
        # Setting time for future savings
        now = datetime.datetime.now().strftime("%m-%d_%H-%M")

        # Retrieve data
        X_full, ts = self.data
        X_full = X_full.transpose(0, 1)
        X_batch = None
        dataset = TensorDataset(X_full)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        data_iterator = iter(dataloader)

        # Sample a brownian motion
        # Just for visualization
        brownian_motion = self.generate_brownian_motion()

        # Start training
        for iteration in tqdm.tqdm(range(self.checkpoint+1, self.n_iters+1)):

            try:
                X_batch = next(data_iterator)[0]
            except StopIteration:
                data_iterator = iter(dataloader)
                X_batch = next(data_iterator)[0]
            X_batch = X_batch.transpose(0, 1)

            self.latent_sde.zero_grad()
            log_p_X, kl = self.latent_sde(X_batch, ts, noise_std = self.noise_std, adjoint = self.adjoint, method = self.method, dt = self.dt)

            loss = - log_p_X + kl * self.kl_scheduler(iteration)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.latent_sde.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()

            if iteration % self.pause_every == 0:
                print(f"Iter: {iteration}/{self.n_iters}, \tLoss: {loss.item():.4f}, \tLog p(X): {log_p_X.item():.4f}, \tKL: {kl.item():.4f}")
                # store checkpoint
                self.model_saver(time=now, checkpoint=True, iteration=iteration)

        if self.save_data:
            self.data_saver(time=now)
        if self.save_model:
            self.model_saver(time=now)
        if self.save_plot:
            self.plot_saver(bm=brownian_motion, time=now, data=X_batch, ts=ts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent SDE Model")
    parser.add_argument("--config", type=str, default="./code/config/config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--train", action='store_true', help="Whether to train the model or just plot the results")
    args = parser.parse_args()

    sde = LatentSDETrainer(config_path=args.config)
    if args.train:
        sde.train()
    else:
        sde.plot()