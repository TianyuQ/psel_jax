# lqrax
JAX-enabled continuous-time LQR solver.

This repo is under active development.

## Install

Follow the [instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) to install JAX before installing this package.

To install: `pip install lqrax`

## Usage

There are two modules: `LQR` and `iLQR`,

The `LQR` module solves the following time-varying LQR problem:

$$
\int_0^T \Big[ (x(t)-\bar{x}(t))^\top Q (x(t)-\bar{x}(t)) + u(t)^\top R u(t) \Big] dt
$$
$$
\text{s.t. } \dot{x}(t) = A(t) x(t) + B(t) u(t), \quad x(0) = x_0
$$

An jupyter notebook example for the `LQR` module is provided [here](examples/lqr_example.ipynb). You can open it in Google Colab [here](https://colab.research.google.com/github/MaxMSun/lqrax/blob/main/examples/lqr_example.ipynb).

The `iLQR` module solves a different time-varying LQR problem:

$$
\int_0^T \Big[ z(t)^\top Q z(t) + v(t)^\top R v(t) + z(t)^\top a(t) + v(t)^\top b(t) \Big] dt
$$
$$
\text{s.t. } \dot{z}(t) = A(t) z(t) + B(t) v(t), \quad z(0) = 0
$$

This formulation is often used as the sub-problem for iterative linear quadratic regulator (iLQR) to calculate the steepest descent direction on the control, where the $z(t)$ and $v(t)$ are perturbations on the system's state and control, and $A(t)$ and $B(t)$ are the linearized system dynamics on the current system trajectory. 

An jupyter notebook example for the `LQR` module is provided [here](examples/ilqr_example.ipynb). You can open it in Google Colab [here](https://colab.research.google.com/github/MaxMSun/lqrax/blob/main/examples/ilqr_example.ipynb).

## Copyright and License

The implementations contained herein are copyright (C) 2024 - 2025 by Max Muchen Sun, and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later). Please see the LICENSE for more information.

If you use the package in your research, please cite this repository. You can see the citation information at the right side panel under "About". The BibTeX file is attached below:
```
@software{sun_lqrax_2025,
    author = {["Sun"], Max Muchen},
    license = {GPL-3.0},
    month = march,
    title = {{lqrax: JAX-enabled continuous-time LQR solver}},
    url = {https://github.com/MaxMSun/lqrax},
    version = {0.0.1},
    year = {2025}
}
```

Contact: msun@u.northwestern.edu
