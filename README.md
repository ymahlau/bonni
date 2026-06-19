![title image](https://github.com/ymahlau/bonni/blob/main/docs/source/_static/bonni.png?raw=true)

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://bonni.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/bonni)](https://pypi.org/project/bonni/)
[![codecov](https://codecov.io/gh/ymahlau/bonni/branch/main/graph/badge.svg)](https://codecov.io/gh/ymahlau/bonni)
[![Continuous integration](https://github.com/ymahlau/bonni/actions/workflows/cicd.yml/badge.svg?branch=main)](https://github.com/ymahlau/bonni/actions/workflows/cicd.yml/badge.svg?branch=main)

# BONNI: Bayesian Optimization via Neural Network surrogates and Interior Point Optimization

BONNI optimizes any black box function WITH gradient information. 
Especially in optimizations with many degree of freedom, gradient-information increases optimization speed. 
In the image, the surrogate fits the function almost perfectly with few observations.

![surrogate image](https://github.com/ymahlau/bonni/blob/main/docs/source/_static/surrogate.png?raw=true)

## Installation

> **Note:** Installation via `pip install bonni` is no longer supported. BONNI depends on
> [cyipopt](https://cyipopt.readthedocs.io), which requires native IPOPT C libraries that
> conda-forge provides but pip does not. Please use [pixi](https://pixi.sh) instead.

Install [pixi](https://pixi.sh/latest/#installation), then clone the repository and run:

```bash
git clone https://github.com/ymahlau/bonni.git
cd bonni
pixi install
```

This resolves all dependencies — including the native IPOPT libraries — from conda-forge automatically.

For GPU-accelerated JAX, add the cuda-enabled jax variant after installation:

```bash
pixi run pip install jax[cuda]
```


## Usage

BONNI provides a nice optimization wrapper similar to the scipy.minimize API:

```python
from bonni import optimize_bonni
from pathlib import Path
import numpy as np

def fn(x: np.ndarray):
    # Input function should return function value and gradient
    value = x[0] ** 2 + x[1]
    grad = np.asarray([2 * x[0], 1])
    return value, grad

xs, ys, gs = optimize_bonni(
    fn=fn,
    bounds=np.asarray([[-1, 1], [0, 1]], dtype=float),
    # BO requires some samples before iterations start. You can either explicitly provide 
    # previous fn evals via `xs=..., ys=..., gs=... or specify a number of random samples. 
    num_bonni_iterations=5,
    num_random_samples=2,
    direction="minimize",
    save_path=Path.cwd(), # save data as npz here
    seed=42,
)
```

Additionally, BONNI includes a convenient wrapper for IPOPT (via [cyipopt](https://cyipopt.readthedocs.io)) shown below:

```python
from bonni import optimize_ipopt
xs, ys, gs = optimize_ipopt(
    fn=fn,
    x0=np.asarray([0.5, 0.5]),  # startpoint of optimization
    bounds=np.asarray([[-1, 1], [0, 1]], dtype=float),
    # IPOPT performs line search each iteration, such that the number 
    # of iterations and fn_eval may not be the same
    max_fn_eval=5,
    max_iterations=3,
    direction="maximize",
    save_path=Path.cwd(),
)
```

## Documentation

You can find the full extensive documentation of BONNI [here](https://bonni.readthedocs.io/en/latest/).

## Examples

### Distributed Bragg Reflector

![dbr image](https://github.com/ymahlau/bonni/blob/main/docs/source/_static/dbr.png?raw=true)

This is a 10d optimization of the layer heights of a distributed Bragg Reflector for color correction in µ-LEDs.
The target spectrum is a step function around 620nm wavelengths.
Compared to other optimization algorithms, BONNI yields the best designs.
For details, we refer to the paper.
The full code for the optimization can be found at `scripts/bragg_reflector.py`.

### Dual-Layer Grating Coupler

![gc image](https://github.com/ymahlau/bonni/blob/main/docs/source/_static/gc.png?raw=true)

This is a 62d optimization of the widths and gap sizes of a dual layer grating coupler.
Compared to other optimization algorithms, BONNI yields the best designs.
For details, we refer to the paper.
The full code for the optimization can be found at `scripts/grating_coupler.py`.

## Citation

If you find this repository helpful for your research, please consider citing:

```
@article{Mahlau_26,
	author = {Yannik Mahlau and Yannick Augenstein and Tyler W. Hughes and Marius Lindauer and Bodo Rosenhahn},
	journal = {Opt. Express},
	number = {13},
	pages = {23160--23174},
	publisher = {Optica Publishing Group},
	title = {Gradient-informed Bayesian and interior point optimization for efficient inverse design in nanophotonics},
	volume = {34},
	month = {Jun},
	year = {2026},
	url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-34-13-23160},
	doi = {10.1364/OE.600198},
}
```


# Other Links

Also check out my other repositories:

> ### 💡 [**fdtdx**](https://github.com/ymahlau/fdtdx) 
> [![GitHub stars](https://img.shields.io/github/stars/ymahlau/fdtdx?style=social)](https://github.com/ymahlau/fdtdx/stargazers) [![GitHub forks](https://img.shields.io/github/forks/ymahlau/fdtdx?style=social)](https://github.com/ymahlau/fdtdx/network/members) 
> 
> A high-performance, differentiable and GPU-accelerated Finite-Difference Time-Domain solver. 

