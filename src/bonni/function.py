from pathlib import Path
from typing import Sequence
import numpy as np
from bonni.constants import INPUT_FN_TYPE


MAX_EVAL_MSG_STR = "Early Stopping IPOPT due to specified maximum fn evaluation"


class FunctionWrapper:
    def __init__(
        self,
        fn: INPUT_FN_TYPE,
        action_bounds: np.ndarray,
        negate: bool,
        max_fn_eval: int | None = None,
        save_path: Path | None = None,
    ):
        self.fn = fn
        self.action_bounds = action_bounds
        self.negate = negate
        self.max_fn_eval = max_fn_eval
        self.save_path = save_path
        
        # lists holding observations
        self.x_list = []
        self.y_list = []
        self.g_list = []
        
        
    def __call__(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert x.ndim == 1
        assert x.shape[0] == self.action_bounds.shape[0]
        assert np.all(x > self.action_bounds[:, 0]) and np.all(x < self.action_bounds[:, 1]), f"Invalid function input: {x}"
        
        if self.max_fn_eval is not None and self.max_fn_eval <= len(self.y_list):
            raise Exception(MAX_EVAL_MSG_STR)
        
        fn_output = self.fn(x)
        
        if not isinstance(fn_output, Sequence):
            raise Exception(f"Invalid function output: {fn_output}")
        
        assert len(fn_output) == 2, f"If function output is sequence, it needs to have length 2, but got {fn_output}"
        y_out, g_out = fn_output[0], fn_output[1]
        assert isinstance(y_out, np.ndarray | float), f"Invalid function output at pos 0: {y_out}"
        assert isinstance(g_out, np.ndarray | float), f"Invalid function output at pos 1: {g_out}"
        y_flat, g_flat = y_out.flatten(), g_out.flatten()
        
        # save function output
        self.x_list.append(x)
        assert y_flat.size == 1, f"Invalid fn output at pos 0: {y_out}"
        y_flat = y_flat[0]
        self.y_list.append(y_flat)
        assert g_flat.size == self.action_bounds.shape[0], f"Invalid fn output at pos 1: {g_out}"
        self.g_list.append(g_flat)
            
        if self.save_path is not None:
            self.append_to_file()
        
        if self.negate:
            y_flat = -y_flat
            g_flat = -g_flat

        return y_flat, g_flat
    
    
    def get_saved_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.asarray(self.x_list, dtype=float)
        ys = np.asarray(self.y_list, dtype=float)
        gs = np.asarray(self.g_list, dtype=float)
        return xs, ys, gs
    
    
    def append_to_file(
        self,
    ):
        assert self.save_path is not None
        xs, ys, gs = self.get_saved_obs()
        np.savez(self.save_path / "data.npz", xs=xs, gs=gs, ys=ys)
        
        