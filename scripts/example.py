import os
from pathlib import Path
import numpy as np
from bonni import optimize_ipopt
from bonni.bonni import optimize_bonni
from bonni.misc import change_to_timestamped_dir
from bonni.synthetic import StyblinskiTangFn

def train_ipopt():
    change_to_timestamped_dir()
    d = 5
    fn = StyblinskiTangFn(d=d)
    x0 = np.zeros((d,), dtype=float)
    optimize_ipopt(
        fn=fn,
        x0=x0,
        bounds=fn.bounds,
        max_fn_eval=5,
        max_iterations=10,
        save_path=Path.cwd(),
    )
    
def train_bonni():
    change_to_timestamped_dir()
    d = 1
    fn = StyblinskiTangFn(d=d)
    optimize_bonni(
        fn=fn,
        bounds=fn.bounds,
        max_fn_evaluations=20,
        max_num_local_samples=3,
        num_random_samples=2,
        direction="maximize",
        save_path=Path.cwd(),
        training_plot_directory=Path.cwd(),
        surrogate_plot_directory=Path.cwd(),
        seed=42,
    )
    
    
def fn(x: np.ndarray):
    # Input function should return function value and gradient
    value = x[0] ** 2 + x[1]
    grad = np.asarray([2 * x[0], 1])
    return value, grad

def optimize_fn():
    change_to_timestamped_dir()
    optimize_bonni(
        fn=fn,
        bounds=np.asarray([[-1, 1], [0, 1]], dtype=float),
        max_fn_evaluations=20,
        num_random_samples=2,
        direction="minimize",
        save_path=Path.cwd(),
        seed=42,
        max_num_local_samples=1,
    )


def optimize_fn_ipopt():
    optimize_ipopt(
        fn=fn,
        x0=np.asarray([0.5, 0.5]),  # startpoint of optimization
        bounds=np.asarray([[-1, 1], [0, 1]], dtype=float),
        # IPOPT performs line search each iteration, such that the number 
        # of iterations and fn_eval may not be the same
        max_fn_eval=10,
        direction="maximize",
        save_path=Path.cwd(),
    )


if __name__ == '__main__':
    train_bonni()
