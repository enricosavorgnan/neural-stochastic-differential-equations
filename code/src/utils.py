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



def plot_1d_latent_sde(X_data, X_samples, plot_path, time, ts, name, n_samples):
    """
    Plots 1D trajectories of the data and the samples from the trained SDE.
    The X-axis represents Time (t), and the Y-axis represents the state value (X_1).
    """
    fig = plt.figure(figsize=(20, 9))
    grid = gs.GridSpec(1, 2)

    # 1D plots use standard 2D axes, so no `projection='3d'` is needed
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])

    # Safely clamp the number of trajectories to the actual batch size
    actual_samples = min(n_samples, X_data.shape[1])

    # Convert the time tensor to a numpy array for the X-axis
    t_np = ts.cpu().numpy()

    # ---------------------------------------------------------
    # Left plot: Data
    # ---------------------------------------------------------
    data_np = X_data.cpu().numpy()

    # Plot trajectories (Time vs State)
    [ax0.plot(t_np, data_np[:, i, 0], alpha=0.8) for i in range(actual_samples)]

    # Scatter the initial starting points (t=0)
    # We use np.repeat to create a time coordinate array [t[0], t[0], ...] matching the sample size
    ax0.scatter(np.repeat(t_np[0], actual_samples), data_np[0, :actual_samples, 0],
                marker='x', color='black', s=50, zorder=5)

    ax0.set_xlabel('Time ($t$)', fontsize=16)
    ax0.set_ylabel('State ($X_1$)', fontsize=16)
    ax0.set_title('Data Trajectories', fontsize=20)
    ax0.grid(True, alpha=0.3)

    # Capture limits to synchronize the right-hand plot
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()

    # ---------------------------------------------------------
    # Right plot: Learned Samples
    # ---------------------------------------------------------
    [ax1.plot(t_np, X_samples[:, i, 0], alpha=0.8) for i in range(actual_samples)]

    ax1.scatter(np.repeat(t_np[0], actual_samples), X_samples[0, :actual_samples, 0],
                marker='x', color='black', s=50, zorder=5)

    ax1.set_xlabel('Time ($t$)', fontsize=16)
    ax1.set_ylabel('State ($X_1$)', fontsize=16)
    ax1.set_title('Learned Trajectories', fontsize=20)
    ax1.grid(True, alpha=0.3)

    # Lock the axes to match the Data plot for accurate visual comparison
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # ---------------------------------------------------------
    # Save Output
    # ---------------------------------------------------------
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"{name}_time_{time}.png"), bbox_inches='tight')
    plt.close()



def plot_2d_latent_sde(X_data, X_samples, plot_path, time, ts, name, n_samples):
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



def plot_3d_latent_sde(X_data, X_samples, plot_path, time, ts, name, n_samples):
    """
    Plots the 3D latent SDE trajectories and saves the plot to the specified path.

    Parameters:
    -----------
    """
    # extract only n_samples trajectories from X_samples
    X_samples = X_samples[:, :n_samples, :]

    fig = plt.figure(figsize=(16, 9), dpi = 600)
    grid = gs.GridSpec(1, 2)
    ax0 = fig.add_subplot(grid[0, 0], projection='3d')
    ax1 = fig.add_subplot(grid[0, 1], projection='3d')

    # Left plot: data.
    z1, z2, z3 = np.split(X_data.cpu().numpy(), indices_or_sections=3, axis=-1)

    [ax0.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(X_data.shape[1])]
    ax0.scatter(z1[0, :X_data.shape[1], 0], z2[0, :X_data.shape[1], 0], z3[0, :X_data.shape[1], 0], marker='x')
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

    [ax1.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(X_samples.shape[1])]
    ax1.scatter(z1[0, :X_samples.shape[1], 0], z2[0, :X_samples.shape[1], 0], z3[0, :X_samples.shape[1], 0], marker='x')
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
