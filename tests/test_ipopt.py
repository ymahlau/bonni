from bonni import optimize_ipopt
import numpy as np


def test_run_ipopt_simple():
    def fn(x: np.ndarray):
        # Input function should return function value and gradient
        value = x[0] ** 2 + x[1]
        grad = np.asarray([2 * x[0], 1])
        return value, grad

    xs, ys, gs = optimize_ipopt(
        fn=fn,
        x0=np.asarray([0.5, 0.5]),  # startpoint of optimization
        bounds=np.asarray([[-1, 1], [0, 1]], dtype=float),
        max_fn_eval=100,
        max_iterations=50,
    )
    assert ys.min() < 0.01


if __name__ == "__main__":
    test_run_ipopt_simple()
