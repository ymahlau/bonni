from bonni import optimize_bonni
import numpy as np


def test_run_bonni_simple():
    def fn(x: np.ndarray):
        # Input function should return function value and gradient
        value = x[0] ** 2 + x[1]
        grad = np.asarray([2 * x[0], 1])
        return value, grad

    # it is too costly to run bonni fully until convergence in a unit test. we only test that no exceptions occur here
    xs, ys, gs = optimize_bonni(
        fn=fn,
        bounds=np.asarray([[-1, 1], [0, 1]], dtype=float),
        num_bonni_iterations=5,
        num_random_samples=10,
        num_iter_until_recompile=5,
    )
