from pathlib import Path
from typing import Literal
import cyipopt
import numpy as np

from bonni.constants import INPUT_FN_TYPE
from bonni.function import MAX_EVAL_MSG_STR, FunctionWrapper


def optimize_ipopt(
    fn: INPUT_FN_TYPE,
    x0: np.ndarray,
    bounds: np.ndarray,
    max_fn_eval: int | None = None,
    max_iterations: int | None = None,
    save_path: Path | str | None = None,
    tol: float = 1e-8,
    direction: Literal["maximize", "minimize"] = "minimize",
    bound_contract_ratio: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize a black-box function using the Interior Point Optimizer (IPOPT).

    This function acts as a convenient wrapper around `cyipopt`, managing the
    interface between the objective function (which must provide gradients) and the
    solver. It handles bound constraints, optimization direction, and recording
    of the optimization history.

    Args:
        fn (Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]): The objective function to optimize.
            It must accept an input array of shape (num_dims,) and return a tuple `(y, g)`,
            where `y` is the scalar objective value and `g` represents gradients of shape (num_dims,).
        x0 (np.ndarray): The initial starting point for the optimization. Must have shape (num_dims,).
        bounds (np.ndarray): A 2D array of shape `(num_dims, 2)` specifying the (lower, upper)
            search space boundaries for each of the `num_dims` input dimensions.
        max_fn_eval (int | None, optional): The maximum number of allowed function evaluations.
            Note that IPOPT performs line searches, so one iteration results in multiple
            function evaluations. Defaults to None.
        max_iterations (int | None, optional): The maximum number of solver iterations.
            Defaults to None.
        save_path (Path | str | None, optional): Directory path to save the results of function
            evaluation as an npz-file. Defaults to None.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-8.
        direction (Literal["maximize", "minimize"], optional): The optimization goal. Defaults to "minimize".
        bound_contract_ratio (float, optional): A small epsilon value used to contract the provided bounds
            slightly (bounds +/- eps * bound_range). This prevents the interior point method from
            starting or evaluating exactly on the boundary, which can cause numerical issues.
            Defaults to 1e-3.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the full history of:
            - `xs`: Input parameters (shape `(N, D)`).
            - `ys`: Objective values (shape `(N,)`).
            - `gs`: Gradients (shape `(N, D)`).
    """

    assert bounds.ndim == 2 and bounds.shape[1] == 2, (
        "bounds needs to be array of shape (degree_of_freedom, 2)"
    )
    assert x0.ndim == 1 and x0.shape[0] == bounds.shape[0], (
        f"x0 needs to have shape (degree_of_freedom,), but got {x0.shape}"
    )
    if direction not in ["maximize", "minimize"]:
        raise Exception(f"Invalid direction argument: {direction}")
    if save_path is not None:
        save_path = Path(save_path)

    wrapper = FunctionWrapper(
        fn,
        np.copy(bounds).astype(float),
        negate=(direction == "maximize"),
        max_fn_eval=max_fn_eval,
        save_path=save_path,
    )

    options = {}
    if max_iterations is not None:
        options["maxiter"] = max_iterations

    bounds_ipopt = np.copy(bounds).astype(float)
    bounds_range = bounds_ipopt[:, 1] - bounds_ipopt[:, 0]
    bounds_eps = bounds_range * bound_contract_ratio / 2
    bounds_ipopt[:, 0] += bounds_eps
    bounds_ipopt[:, 1] -= bounds_eps

    try:
        cyipopt.minimize_ipopt(
            fun=wrapper,
            x0=x0,
            args=(),
            method=None,
            jac=True,
            bounds=bounds_ipopt,
            tol=tol,
            options=options,
        )
    except Exception as e:
        # if we are early stopping due, then do nothing. If this is real error, raise it again
        if str(e) != MAX_EVAL_MSG_STR:
            raise e

    return wrapper.get_saved_obs()
