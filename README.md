# Neural Stochastic Differential Equations

The repository is intended to provide code examples, papers and a summary about Neural SDE.


## Introduction

### Neural and Latent SDEs 
In a glimpse, **Neural Stochastic Differential Equations** (Neural SDE) consist in SDE where both the drift and the diffusion term are parameterized by Neural Networks. \
Formally, they can be seen as a *Deep Gaussian Latent Model* (DGLM) at the diffusion limit, i.e., with infinitely deep and having the maps from each layer to the following close to 0 (see next sections for more information). \
Neural SDEs can be used as *continuisations* of DGLM, thus having a practical application in the modeling of time series and, clearly, of stochastic models. 

The repository focuses on a particular application of Neural SDE, called **Latent SDE**, where the expressive power of SDE is used to learn (latent) continuous-time dynamics. \
Both observed and latent variables are supposed to be subjected to white noise. 

The model is actually a *Variational AutoEncoder* (VAE). \
The encoder takes in input a whole *flipped* (from the end to the beginning) trajectory of the observed variables and extracts the information about it exploiting GRU layers (light-weight Recurrent Neural Networks). \
The information is then used to reconstruct the mean and the variance of the prior gaussian distribution.
Stochastic Adjoint methods are then run to compute the gradients of the loss function, which is the Evidence Lower Bound (ELBO) of the model. \
Adjoint methods also allow the computation of a latent variables' trajectory, which is then used to reconstruct the observed variable by feeding the decoder module. \
Adjoints also compute the (path-wise) KL divergence (*Radon-Nikodym derivative*) between the paths of the posterior SDE and the prior SDE; this divergence is then added to the KL divergence computed between the prior and posterior distributions in order to compute the loss. \

The loss is parameterized by both the parameters of the prior distribution, of the posterior one and by the parameters of the SDEs deriving by the use of adjoints methods.

### (Theory of) SDEs as Diffusion Limits of DGLMs

### Stochastic Adjoint Methods
Stochastic adjoint methods have been developed in *Li et al.*, *2020*, and are a stochastic extension of the ordinary adjoint methods used to solve Neural and Latent ODEs. \
The deterministic version consisted into the integration of a backward ODE, so to efficiently compute the gradients of loss with respect to the neural network and ODEs parameters.

The stochastic version is, at the core, pretty similar. \
It exploits the symmetry of a Stratonovich SDE to compute a *backward* flow, i.e., a (stochastic) path from the end to the beginning point. \
This inverse path is crucial since it allows the computation of the gradient of the loss w.r.t. the SDE parameters by integrating a new Stratonovich SDE. \
To visualize the adjoint system solved, let's write down a bit of equations:
- **Forward SDE**, the SDE ruling the observations:
$$
  X_t:= \phi_{s,t}(x_s) = x_s + \int_s^t \mu(x_q, q)dq + \int_s^t \sigma(x_q, q) \circ dW_q,
$$
  where $\mu$ represents the drift term, $\sigma$ the diffusion term, and $\{dW_q\}_{q=s:t}$ a realization of a Wiener process.
- **Backward SDE**, valid in case of a (simple) invertibility condition:
$$
  \psi_{s,t}(x_t) = x_t - \int_s^t \mu(\psi_{q,t}(x_t), q)dq - \int_s^t \sigma(\psi_{q,t}(x_t), q) \circ dY_q,
$$
  where here $\{dY_q\}_{q=t:s}$ is exactly the inverted Wiener process previously identified as $\{dW_q\}_{q=s:t}$.

Now we want to derive closed forms for the derivatives of the processes $\phi$ and $\psi$ with respect to the variables. Since Stratonovich SDEs allow the use of common calculus to compute the chain rule, we get:
$$
\begin{align}
  J_{s,t}(x)&:= \nabla_x \psi_{s,t}(x) = \mathbb{I}^d - \int_s^t \nabla \mu(\psi_{q,t}(x), q)J_{q,t}(x)dq - \int_s^t \nabla \sigma(\psi_{q,t}(x), q)J_{s,t}(x) \circ dY_q \\
  K_{s,t}(x)&:= J_{s,t}(x)^{-1} = \mathbb{I}^d + \int_s^t K_{q,t}(x) \nabla \mu(\psi_{q,t}(x), q) dq + \int_s^t K_{q,t}(x) \nabla \sigma(\psi_{q,t}(x), q) \circ dY_q
\end{align}
$$
However, since the endpoint is non-deterministic, we need to compute the gradient by taking into account that the loss $\mathcal{L}$ is function of the stochastic process. \
We define $A_{s,t}(x) = \partial \mathcal{L}(\phi_{s,t}(x)) / \partial x$ and, applying the chain rule, $A_{s,t}(x) = \nabla \mathcal{L}(\phi_{s,t}(x))\nabla \phi_{s,t}(x)$. \
Analogously, we define $B_{s,t}(x):=A_{s,t}(\psi_{s,t}(x))$. With a bit of calculus, it turns out that:
$$
\begin{align}
  B_{s,t}(x)  &= \nabla \mathcal{L} \cdot K_{s,t}(x) \\ 
              &= \nabla \mathcal{L}(x) + \int_s^t \nabla \mu(\psi_{q,t}(x),q)^T B_{q,t}(x)dq + \int_s^t \nabla \sigma(\psi_{q,t}(x), x)^T B_{q,t}(x) \circ dY_q
\end{align}
$$
This very last equation, with the backward SDE we defined above, constitutes the **adjoint system**, whose solution is the goal of stochastic adjoint methods. \

The target is approximated by using two (deterministic) maps. \
The forward pass is mapped by a function $G(x, \{W\}; \theta) \approx \phi_{0, T}(z) $, while the solution to the backward SDE is computed by a function $F(\phi, \{W\}; \omega) \approx B_{0, T}(x)$ so that $F(\phi, \{W\}; \omega) \approx F(G(x), \{W\})$.


## `code/` Structure
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






## References

- The main paper references are:
  - **Tzen, Raginsky**, *Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit*, 2019
  - **Li et al.**, *Scalable Gradients for Stochastic Differential Equations*, 2020
  - **Kidger**, *On Neural Differential Equations*, 2022

  Other relevant papers are shared in the `docs/` folder.


- Code is heavily inspired by the examples in the [torchsde library](https://github.com/google-research/torchsde).
  However, much of the implementation has been rewritten so to be more accessible and useful for educational purposes.
  Moreover, the vast majority of the examples has been entirely written from scratch.


- Lastly, an excellent introduction to LatentSDEs is provided in [this video lecture](https://youtu.be/EAsXp8NaCR8?si=dFOazGxnO5RAPV8l) by **Duvenaud** at the Toronto ML Summit.
  


