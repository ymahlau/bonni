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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert bounds.ndim == 2 and bounds.shape[1] == 2, "bounds needs to be array of shape (degree_of_freedom, 2)"
    assert x0.ndim == 1 and x0.shape[0] == bounds.shape[0], f"x0 needs to have shape (degree_of_freedom,), but got {x0.shape}"
    if direction not in ["maximize", "minimize"]:
        raise Exception(f"Invalid direction argument: {direction}")
    if save_path is not None:
        save_path = Path(save_path)
    
    wrapper = FunctionWrapper(
        fn, 
        bounds, 
        negate=(direction=="maximize"),
        max_fn_eval=max_fn_eval,
        save_path=save_path,
    )
    
    options = {}
    if max_iterations is not None:
        options['maxiter'] = max_iterations
    
    try:
        cyipopt.minimize_ipopt(
            fun=wrapper,
            x0=x0,
            args=(),
            method=None,
            jac=True,
            bounds=bounds,
            tol=tol,
            options=options,
        )
    except Exception as e:
        # if we are early stopping due, then do nothing. If this is real error, raise it again
        if str(e) != MAX_EVAL_MSG_STR:
            raise e
    
    return wrapper.get_saved_obs()
    
    
    
    
    
    
