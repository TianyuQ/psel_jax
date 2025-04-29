<div align="center">
<img src="https://raw.githubusercontent.com/MaxMSun/lqrax/main/media/lqrax_logo.png" alt="logo" width="150"></img>
</div>

# LQRax
LQRax is [JAX](https://github.com/jax-ml/jax)-enabled continuous-time LQR solver. It is essentially a Riccati equation solver completely written in JAX:

- It accelerates numerical simulation through JAX's [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) mechanism;
- It allows rapid prototyping of iterative LQR (iLQR) for nonlinear control by supporting [auto-differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) on the loss function and dynamics;
- It also enables batch-based large-scale optimal control using JAX's [`vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) mechanism.

*This repo is currently under active development.*

<div align="center">
<img src="https://raw.githubusercontent.com/MaxMSun/lqrax/main/media/drone_landing.gif" alt="example" width="600"></img>
</div>


## Install

Follow the [instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) to install JAX before installing this package.

To install: `pip install lqrax`

## Usage

There are two modules: `LQR` and `iLQR`,

The `LQR` module solves the following time-varying LQR problem:

$$
\arg\min_{u(t)} \int_0^T \Big[ (x(t)-x_{ref}(t))^\top Q (x(t)-x_{ref}(t)) + u(t)^\top R u(t) \Big] dt
$$
$$
\text{s.t. } \dot{x}(t) = A(t) x(t) + B(t) u(t), \quad x(0) = x_0
$$

An jupyter notebook example for the `LQR` module is provided [here](https://github.com/MaxMSun/lqrax/blob/main/examples/lqr_example.ipynb). You can open it in Google Colab [here](https://colab.research.google.com/github/MaxMSun/lqrax/blob/main/examples/lqr_example.ipynb).

The `iLQR` module solves a different time-varying LQR problem:

$$
\arg\min_{v(t)} \int_0^T \Big[ z(t)^\top Q z(t) + v(t)^\top R v(t) + z(t)^\top a(t) + v(t)^\top b(t) \Big] dt
$$
$$
\text{s.t. } \dot{z}(t) = A(t) z(t) + B(t) v(t), \quad z(0) = 0.
$$

This formulation is often used as the sub-problem for iterative linear quadratic regulator (iLQR) to calculate the steepest descent direction on the control for a general nonlinear control problem:

$$
\arg\min_{u(t)} \int_0^T l(x(t), u(t)) dt, \text{ s.t. } \dot{x}(t) = f(x(t), u(t)),
$$ 

where the $z(t)$ and $v(t)$ are perturbations on the system's state $x(t)$ and control $u(t)$, and $A(t)$ and $B(t)$ are the linearized system dynamics $f(x(t), u(t))$ on the current system trajectory with respect to the state and control. 

An jupyter notebook example of using the `iLQR` module for a nonlinear control problem is provided [here](https://github.com/MaxMSun/lqrax/blob/main/examples/ilqr_example.ipynb). You can open it in Google Colab [here](https://colab.research.google.com/github/MaxMSun/lqrax/blob/main/examples/ilqr_example.ipynb).

## Copyright and License

The implementations contained herein are copyright (C) 2024 - 2025 by Max Muchen Sun, and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later). Please see the LICENSE for more information.

If you use the package in your research, please cite this repository. You can see the citation information at the right side panel under "About". The BibTeX file is attached below:
```
@software{sun_lqrax_2025,
    author = {["Sun"], Max Muchen},
    license = {GPL-3.0},
    month = march,
    title = {{LQRax: JAX-enabled continuous-time LQR solver}},
    url = {https://github.com/MaxMSun/lqrax},
    version = {0.0.5},
    year = {2025}
}
```

Contact: msun@u.northwestern.edu
