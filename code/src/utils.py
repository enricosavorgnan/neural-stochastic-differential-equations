import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

import os
import yaml
import time
import argparse

from typing import Callable, Sequence

import torchsde


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
