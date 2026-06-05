import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os



def get_sde_kwargs(config, approx : bool):
    """
    Reconstructs the SDE kwargs from the config dictionary.
    If approx is True, it uses the 'fixed_params' from the config, otherwise it uses the 'params_sample_type' to sample the parameters.

    Parameters:
    -----------
    config  : dict
        The configuration dictionary containing the SDE parameters.
    approx  : bool, optional
        Whether to use the 'fixed_params' from the config or to sample the parameters based on

    Returns:
    --------
    sde_kwargs : dict
        A dictionary containing the SDE parameters to be passed to the
    """
    sde_kwargs = {
        'n_params': config['sde']['n_params'],
        'noise_type': config['sde']['noise_type'],
        'sde_type': config['sde']['sde_type'],
        't_span': config['sde']['t_span'],
        't_size': config['sde']['t_size'],
        'params_sample_type': config['sde']['params_sample_type'] if approx else 'fixed',
        'fixed_params': config['sde']['fixed_params'],
        'method': config['sde']['method']
    }
    return sde_kwargs


def plot_1d_latent_sde(X_data, X_samples, plot_path, time, ts, name):
    # call plot_2d_latent_sde with the first two dimensions of the data and samples
    plot_2d_latent_sde(X_data[:, :, :2], X_samples[:, :, :2], plot_path, time, ts, name)


def plot_2d_latent_sde(X_data, X_samples, plot_path, time, ts, name):
    """
    Plots the 2D latent SDE trajectories and saves the plot to the specified path.

    Parameters:
    -----------
    X_data : torch.Tensor
        The original data trajectories.
    X_samples : torch.Tensor
        The sampled trajectories from the learned model.
    plot_path : str
        The directory path where the plot will be saved.
    time : int
        The current time step or epoch for labeling the plot.
    ts : torch.Tensor
        The time steps corresponding to the trajectories.
    name : str
        A name identifier for the plot file.

    Returns:
    --------
    None
        Saves the plot as a PNG file in the specified directory.
    """
    fig = plt.figure(figsize=(16, 9), dpi = 600)
    grid = gs.GridSpec(1, 2)
    ax0 = fig.add_subplot(grid[0, 0], projection='2d')
    ax1 = fig.add_subplot(grid[0, 1], projection='2d')

    # Left plot: data.
    z1, z2 = np.split(X_data.cpu().numpy(), indices_or_sections=3, axis=-1)
    [ax0.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(len(X_samples))]
    ax0.scatter(z1[0, :len(X_samples), 0], z2[0, :len(X_samples), 0], marker='x')
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])

    ax0.set_xlabel('$X_1$', labelpad=0., fontsize=16)
    ax0.set_ylabel('$X_2$', labelpad=.5, fontsize=16)
    ax0.set_title('Data Trajectory', fontsize=20)
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()

    # Right plot: samples from learned model.
    z1, z2 = np.split(X_samples, indices_or_sections=3, axis=-1)

    [ax1.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(len(X_samples))]
    ax1.scatter(z1[0, :len(X_samples), 0], z2[0, :len(X_samples), 0], marker='x')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xlabel('$X_1$', labelpad=0., fontsize=16)
    ax1.set_ylabel('$X_2$', labelpad=.5, fontsize=16)
    ax1.set_title('Samples', fontsize=20)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    image_save_path = os.path.join(plot_path, f"plot_{name}_time_{time}.png")
    plt.savefig(image_save_path)
    plt.close()



def plot_3d_latent_sde(X_data, X_samples, plot_path, time, ts, name):
    """
    Plots the 3D latent SDE trajectories and saves the plot to the specified path.

    Parameters:
    -----------
    """
    fig = plt.figure(figsize=(16, 9), dpi = 600)
    grid = gs.GridSpec(1, 2)
    ax0 = fig.add_subplot(grid[0, 0], projection='3d')
    ax1 = fig.add_subplot(grid[0, 1], projection='3d')

    # Left plot: data.
    z1, z2, z3 = np.split(X_data.cpu().numpy(), indices_or_sections=3, axis=-1)
    [ax0.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(len(X_samples))]
    ax0.scatter(z1[0, :len(X_samples), 0], z2[0, :len(X_samples), 0], z3[0, :10, 0], marker='x')
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.set_zticklabels([])

    ax0.set_xlabel('$X_1$', labelpad=0., fontsize=16)
    ax0.set_ylabel('$X_2$', labelpad=.5, fontsize=16)
    ax0.set_zlabel('$X_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax0.set_title('Data Trajectory', fontsize=20)
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    zlim = ax0.get_zlim()

    # Right plot: samples from learned model.
    z1, z2, z3 = np.split(X_samples, indices_or_sections=3, axis=-1)

    [ax1.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(len(X_samples))]
    ax1.scatter(z1[0, :len(X_samples), 0], z2[0, :len(X_samples), 0], z3[0, :10, 0], marker='x')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xlabel('$X_1$', labelpad=0., fontsize=16)
    ax1.set_ylabel('$X_2$', labelpad=.5, fontsize=16)
    ax1.set_zlabel('$X_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax1.set_title('Samples', fontsize=20)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)

    image_save_path = os.path.join(plot_path, f"plot_{name}_time_{time}.png")
    plt.savefig(image_save_path)
    plt.close()
