![title image](https://github.com/ymahlau/bonni/blob/main/img/bonni.png?raw=true)

# BONNI: Bayesian Optimization via Neural Network surrogates and Interior Point Optimization

BONNI optimizes any black box function WITH gradient information. 
Especially in optimizations with many degree of freedom, gradient-information, gradient-information increases optimization speed. 
In the image, the surrogate fits the function almost perfectly with few observations.

![surrogate image](https://github.com/ymahlau/bonni/blob/main/img/surrogate.png?raw=true)

## Installation

You can install BONNI simply via `pip install bonni`

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

Additionally, BONNI includes a convenient wrapper for IPOPT, which can be difficult to install / use:

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
    save_path=Path.cwd(),
)
```


## Citation

If you find this repository helpful for your research, please consider citing:

TODO insert citation as soon as paper online.
