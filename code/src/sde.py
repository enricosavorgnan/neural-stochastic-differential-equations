"""
This file implements the core structure for the experiment at section 7.1 of Li et al. 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

import os
import yaml
import datetime
import argparse

from typing import Callable, Sequence

import torchsde

from utils import get_sde_kwargs





class SDE(nn.Module):
    """
    This class implements the structure for the SDE,
    including the drift and diffusion functions.
    """

    def __init__(self,
                 f : Callable[[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor], float],
                 g : Callable[[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor], float],
                 n_params : int = 1,
                 params_sample_type : str = 'gaussian',
                 noise_type : str = 'diagonal',
                 sde_type : str = 'ito',
                 **kwargs):

        super().__init__()
        assert params_sample_type in ['gaussian', 'uniform', 'fixed'], f"Invalid params_sample_type: {params_sample_type}"
        assert noise_type in ['diagonal', 'full'], f"Invalid noise_type: {noise_type}"
        assert sde_type in ['ito', 'stratonovich'], f"Invalid sde_type: {sde_type}"

        if params_sample_type == 'gaussian':
            self.params = torch.randn(n_params)
            # make sure params are positive
            for i in range(n_params):
                if self.params[i] <= 0:
                    self.params[i] *= -1
        elif params_sample_type == 'uniform':
            self.params = torch.rand(n_params)
        elif params_sample_type == 'fixed':
            self.params = torch.tensor( kwargs.get('fixed_params', np.zeros(n_params)) )
        self.params = nn.Parameter(self.params)
        self.noise_type = noise_type
        self.sde_type = sde_type

        self.drift_func = f
        self.diffusion_func = g


    def drift(self, t, x):
        return self.drift_func(t, x, self.params)


    def diffusion(self, t, x):
        return self.diffusion_func(t, x, self.params)



class SDESolver:
    """
    This class implements the structure for solving the SDE and training its parameters
    using either the direct method or the adjoint method.
    """

    def __init__(self,
                 t_span : list[ float | int],
                 t_size : int = 100,
                 batch_size : int = 1,
                 state_size : int = 1,
                 device : str = 'cpu',
                 **kwargs
                 ):
        self.t_span = t_span
        self.t_size = t_size
        self.batch_size = batch_size
        self.state_size = state_size
        self.device = device
        self.sde = SDE(
                       f=kwargs.get('f', lambda t, x: - t * x),
                       g=kwargs.get('g', lambda t, x: 0.1 * t * x),
                       n_params=kwargs.get('n_params', 1),
                       params_sample_type=kwargs.get('params_sample_type', 'gaussian'),
                       noise_type=kwargs.get('noise_type', 'diagonal'),
                       sde_type=kwargs.get('sde_type', 'ito'),
                       fixed_params=kwargs.get('fixed_params', None)
                       ).to(self.device)

        self.Ts = torch.linspace(torch.tensor(t_span[0]), torch.tensor(t_span[1]), int(t_size)).to(self.device)
        self.X0 = torch.full(size=(self.batch_size, state_size), fill_value=0.1).to(device)          # 0.1 works fine in most cases
        self.brownian = None


    def trajectory(self, adjoint : bool = True, method : str = 'euler', **kwargs):
        """
        Computes the trajectory of the SDE solution with respect to the parameters
        using either the direct method or the adjoint method.

        Parameters:
        -----------
        adjoint : bool
             If True, uses the adjoint method to compute the gradient.
             If False, uses the direct method.
        method : str
             The numerical method to use for solving the SDE.
             Valid options are 'euler' and 'milstein'.

        Returns:
        --------
        Xs : torch.Tensor
             The solution of the SDE at the specified time points.
        """
        assert method in ['euler', 'milstein', 'euler_heun', 'midpoint'], f"Invalid method: {method}"
        if kwargs.get('fixed_noise', False):
            seed = int(kwargs.get('seed', 42))
            torch.manual_seed(seed)
            self.compute_noise_trajectory(seed=seed)

        if adjoint:
            Xs = self.compute_adjoint_trajectory(method, **kwargs)
        else:
            Xs = self.compute_trajectory(method, **kwargs)

        return Xs


    def compute_noise_trajectory(self, **kwargs):
        """
        Computes the noise trajectory for the SDE.
        """
        self.brownian = torchsde.BrownianInterval(t0=self.t_span[0],
                                                  t1=self.t_span[1],
                                                  size=(self.batch_size, self.state_size),
                                                  device=self.device,
                                                  **kwargs)


    def compute_trajectory(self, method, **kwargs):
        """
        Computes the gradient using the direct method.

        Parameters:
        -----------
        method : str
             The numerical method to use for solving the SDE.
             Valid options are 'euler' and 'milstein'.

        Returns:
        --------
        Xs : torch.Tensor
             The solution of the SDE at the specified time points.
        """
        Xs = torchsde.sdeint(sde = self.sde,
                             y0 = self.X0,
                             ts = self.Ts,
                             method=method, names={"drift": "drift", "diffusion": "diffusion"},
                             bm = self.brownian if self.brownian else None)
        return Xs


    def compute_adjoint_trajectory(self, method, **kwargs):
        """
        Computes the gradient using the adjoint method.

        Parameters:
        -----------
        method : str
             The numerical method to use for solving the SDE.
             Valid options are 'euler' and 'milstein'.

        Returns:
        --------
        Xs : torch.Tensor
             The solution of the SDE at the specified time points.
        """
        Xs = torchsde.sdeint_adjoint(sde = self.sde,
                                     y0 = self.X0,
                                     ts = self.Ts,
                                     method=method,
                                     names={"drift": "drift", "diffusion": "diffusion"},
                                     bm = kwargs.get('noise', None)
                                                        if kwargs.get('fixed_noise', False) else None)
        return Xs


    def plot(self, ts, samples : torch.Tensor, xlabel, ylabel, title=''):
        ts = ts.cpu()
        samples = samples.squeeze().cpu()

        fig = plt.figure(dpi=600)
        for i, sample in enumerate(samples):
            label = 'Target' if i == len(samples) - 1 else f'Iteration {i}'
            plt.plot(ts, sample, marker='x', label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        return fig


    def train(self,
              loss : Callable[[float | torch.Tensor | Sequence[torch.Tensor], float | torch.Tensor | Sequence[torch.Tensor]], torch.Tensor],
              optimizer : torch.optim.Optimizer,
              target_trajectory : torch.Tensor | Sequence[torch.Tensor],
              device : str = 'cpu',
              n_iters : int = 100,
              adjoint : bool = True,
              **kwargs
              ):
        """
        Trains the SDE parameters using either the direct method or the adjoint method.

        Parameters:
        -----------
        loss : Callable[[float | torch.Tensor, float | torch.Tensor], float | torch.Tensor]
             The loss function to optimize.
             It should take the SDE solution and the parameters as input and return a scalar loss.
        n_iters : int
             The number of training iterations.
        adjoint : bool
             If True, uses the adjoint method to compute the gradient.
             If False, uses the direct method.

        Returns:
        --------
        loss_value : float
             The value of the loss function after training.
        params_grad : torch.Tensor
             The gradient of the loss function with respect to the SDE parameters.
        """
        optim = optimizer if optimizer else torch.optim.Adam(self.sde.parameters(), lr=0.01)
        loss_value = torch.tensor(0.0)

        target_trajectory = target_trajectory.to(device)
        trajectories = torch.empty((n_iters+1, self.t_size, self.batch_size, self.state_size))

        print(f"Iteration: 0/{n_iters}, \t"
              f"Loss: ND, \t"
              f"Params: {self.sde.params.data.cpu().numpy()}")

        for _iter in range(1, n_iters+1):
            optim.zero_grad()
            Xs = self.trajectory(adjoint=adjoint, **kwargs)

            loss_value = loss(Xs, target_trajectory)
            loss_value.backward()
            optim.step()

            trajectories[_iter] = Xs.detach().cpu()

            print(f"Iteration: {_iter}/{n_iters}, \t"
                  f"Loss: {loss_value.item():.4f}, \t"
                  f"Params: {self.sde.params.data.cpu().numpy()}")

        params = self.sde.params.data.clone().detach()

        if kwargs.get('plot', False):
            # merge trajectories and target trajectory for plotting
            trajectories = torch.cat((trajectories, target_trajectory.unsqueeze(0)), dim=0)
            # remove initial trajectory (iteration 0) for better visualization
            trajectories = trajectories[1:]

            fig = self.plot(self.Ts, trajectories, xlabel='Time', ylabel='State', title='Trajectory')

            fig.show()

            if kwargs.get('save', False):
                now = datetime.datetime.now().strftime("%m-%d_%H-%M")
                true_params = kwargs.get('true_params', 'unknown')
                save_path = kwargs.get('path', '.')
                def _label(value):
                    if isinstance(value, str):
                        return value
                    array = np.asarray(value)
                    if array.ndim == 0:
                        item = array.item()
                        try:
                            return format(item, ".4f")
                        except (TypeError, ValueError):
                            return str(item)
                    return "_".join(format(item, ".4f") for item in array.reshape(-1))

                save_tag = (
                    f"true_params_{_label(true_params)}__"
                    f"learned_params_{_label(params.cpu().numpy())}__"
                    f"n_iters_{n_iters}__"
                    f"method_{kwargs.get('method', 'unknown')}__"
                    f"lr_{optim.param_groups[0]['lr']}__"
                    f"time_{now}.png"
                )
                path = os.path.join(save_path, save_tag)
                # os.makedirs(path, exist_ok=True)
                fig.savefig(path, dpi=600)

        return loss_value.item(), params




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDE Experiment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    eval_namespace = {"torch": torch, "nn": nn}
    f = eval("lambda t,x,p: " + config['sde']['drift'], eval_namespace)
    g = eval("lambda t,x,p: " + config['sde']['diffusion'], eval_namespace)
    loss = eval(config['training']['loss'], eval_namespace)
    device = config['training']['device']

    kwargs = get_sde_kwargs(config=config, approx=False)
    sde_solver = SDESolver(**kwargs)


    true_solver = SDESolver(f=f, g=g, **kwargs)
    with torch.no_grad():
        target_Xs = true_solver.trajectory(adjoint=False, method = config['sde']['method'])
    true_params = true_solver.sde.params.data.clone().detach()
    # print(f" Target Trajectory: {target_Xs.squeeze().cpu().numpy()} \t True Params: {true_params}")
    # print("\nStarting Optimization via Stochastic Adjoint Method...")


    kwargs = get_sde_kwargs(config=config, approx=True)
    learnable_solver = SDESolver(f=f, g=g, device=device, **kwargs)
    optimizer = eval(config['training']['optimizer'],
                     globals=eval_namespace,
                     locals = {"params": learnable_solver.sde.parameters(),
                               "lr": config['training']['lr']}
                     )

    loss, params = learnable_solver.train(loss=loss,
                           optimizer= optimizer,
                           target_trajectory=target_Xs,
                           n_iters=config['training']['n_iters'],
                           plot=config['training']['plot'],
                           device = config['training']['device'],
                           save = config['training']['save'],
                           path = config['training']['save_path'],
                           true_params = true_params,
                           **kwargs)

    print(f"\nFinal Loss: {loss:.4f}, \t Final Params: {params.cpu().numpy()} \t True Params: {true_params}")




