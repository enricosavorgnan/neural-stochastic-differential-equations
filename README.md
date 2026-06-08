# Neural Stochastic Differential Equations

The repository is intended to provide code examples, papers and a summary about Neural SDE. 


### Code Structure
```aiignore
├── docs/                  # Papers, slides and other resources
├── code/                  # Code
    ├── config/            # Config YAML files
    ├── data/              # Data sampled
    ├── models/            # Models
    ├── img/               # Images
    └── src/               # Source code
        ├── plain_sde.py
        ├── latent_sde.py
        ├── utils.py
        └── systems.py
```

- `plain_sde.py`: \
  Contains a simple implementation of an `SDE` class to define a Stochastic Differential Equation in a way which is compatible with `torchsde` library.
  `torchsde` takes care of solving the equation by applying several algorithms, including *Euler-Maruyama*, *Milstein*, *Euler-Heun*, *Midpoints* and others.
  The file also contains an helper method `SDESolver` which, given a valid configuration YAML file computes and plots trajectories for a given `SDE` instance, and solve it using both adjoint methods or common solvers.

  A valid YAML file is like `./code/config/config_ou.yaml`, which instantiate an Ornstein-Uhlenbeck process.
  Plots are stored in `./code/img/` folder.

  **N.B.**: the scope of this file is just to provide evidences that the adjoint method is able to reconstruct an SDE. It actually *does NOT learn* anything interesting!

- `latent_sde.py`: \
  Here is where things become intriguing. 
  The file implements a Latent Stochastic Differential Equation, i.e., a Deep Latent Gaussian Model aiming to reconstruct drift and diffusion processes of given SDE. 
  Some examples of valid SDE are implemented in `systems.py` file, as we will discuss below.

  As discussed in *Li et al., 2019*, the DLGM trains *jointly* the prior and the posterior SDEs, allowing to write a close form for the ELBO, i.e., for the loss to optimize.

  A class `LatentSDETrainer` allows to efficiently train the model, loading and generating data and brownian noise, plot trajectories. It requires a configuration YAML file, as in `./code/config/config_lorenz.yaml` where a Lorenz Attractor is instantiated.

  Models are allowed to run on CUDA or XPU devices.

- `systems.py`: \ 
  Contains different SDE models ready-to-use for training DLGMs, including:
  - **Lorenz Attractor**
  - **Climate Model** by Bezzi, Sutara, Vulpiani, Parisi, 1983






### References

- The main paper references are:
  - **Tzen, Raginsky**, *Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit*, 2019
  - **Li et al.**, *Scalable Gradients for Stochastic Differential Equations*, 2020
  - **Kidger**, *On Neural Differential Equations*, 2022

  Other relevant papers are shared in the `docs/` folder.


- Code is heavily inspired by the examples in the [torchsde library](https://github.com/google-research/torchsde).
  However, much of the implementation has been rewritten so to be more accessible and useful for educational purposes.
  Moreover, the vast majority of the examples has been entirely written from scratch.


- Lastly, an excellent introduction to LatentSDEs is provided in [this video lecture](https://youtu.be/EAsXp8NaCR8?si=dFOazGxnO5RAPV8l) by **Duvenaud** at the Toronto ML Summit.
  


