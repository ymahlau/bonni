![title image](https://github.com/ymahlau/bonni/blob/main/docs/source/_static/bonni.png?raw=true)

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://bonni.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/bonni)](https://pypi.org/project/bonni/)

# BONNI: Bayesian Optimization via Neural Network surrogates and Interior Point Optimization

BONNI optimizes any black box function WITH gradient information. 
Especially in optimizations with many degree of freedom, gradient-information increases optimization speed. 
In the image, the surrogate fits the function almost perfectly with few observations.

![surrogate image](https://github.com/ymahlau/bonni/blob/main/docs/source/_static/surrogate.png?raw=true)

## Installation

You can install BONNI simply via 

```bash
pip install bonni
```
We recommend installing also the GPU-acceleration from JAX, which will massively increase speed:
```bash
pip install jax[cuda]
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

Additionally, BONNI includes a convenient wrapper for IPOPT. The standard IPOPT package can be difficult to install/use, so we created a convenient wrapper shown below:

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

TODO insert citation as soon as paper online.
