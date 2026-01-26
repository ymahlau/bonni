from pathlib import Path
from typing import Literal
from bonni.acquisition.ei import EIConfig
import jax
import numpy as np

from bonni.bo import bo_loop
from bonni.constants import INPUT_FN_TYPE
from bonni.function import FunctionWrapper
from bonni.model.ensemble import MLPEnsembleConfig
from bonni.model.mlp import MLPModelConfig
from bonni.model.optim import OptimConfig


def optimize_bonni(
    fn: INPUT_FN_TYPE,
    bounds: jax.Array | np.ndarray,
    num_bonni_iterations: int,
    num_random_samples: int | None = None,
    direction: Literal["maximize", "minimize"] = "minimize",
    seed: int | jax.Array | None = None,
    save_path: Path | str | None = None,
    xs: np.ndarray | jax.Array | None = None,
    ys: np.ndarray | jax.Array | None = None,
    gs: np.ndarray | jax.Array | None = None,
    num_iter_until_recompile: int = 50,
    ensemble_size: int = 20,
    training_plot_directory: Path | None = None,
    surrogate_plot_directory: Path | None = None,
    num_acq_optim_samples: int = 100,
    custom_ei_config: EIConfig | None = None,
    custom_optim_config: OptimConfig | None = None,
    custom_base_model_config: MLPModelConfig | None = None,
    num_embedding_channels: int = 1,
    non_diff_params: np.ndarray | None = None,
    num_acq_optim_runs: int = 5,
    num_initial_acq_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize any black-box function with BONNI.
    Executes the Bayesian Optimization (BO) loop using an MLP Ensemble surrogate model.

    Args:
        fn (Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]): The black-box objective function to optimize. It must accept
            an input array of shape (D,) and return a tuple `(y, g)`, where `y` is the scalar objective
            and `g` represents gradients or auxiliary outputs of shape (D,). D represents the number of dimensions.
        bounds (jax.Array | np.ndarray): A 2D array of shape `(D, 2)` specifying the
            (lower, upper) search space boundaries for each of the `D` input dimensions.
        num_bonni_iterations (int): The number of Bayesian Optimization steps (active
            learning iterations) to perform after initialization. Must be > 0.
        num_random_samples (int | None, optional): The number of initial random samples to
            evaluate before starting the BO loop. If `xs`/`ys`/`gs` are not provided, this
            must be specified and non-zero. Defaults to None.
        direction (Literal["maximize", "minimize"], optional): The optimization goal.
            If "minimize", the wrapper internally negates the objective values.
            Defaults to "minimize".
        seed (int | jax.Array | None, optional): Random seed or JAX PRNGKey for reproducibility.
            If None, a random seed is generated. Defaults to None.
        save_path (Path | str | None, optional): Directory path to save the results of function evaluation as npz-file. 
            Defaults to None.
        xs (np.ndarray | jax.Array | None, optional): Existing history of input points.
            Must be provided if `ys` and `gs` are provided. Defaults to None.
        ys (np.ndarray | jax.Array | None, optional): Existing history of objective values. Must be provided 
            if `xs` and `gs` are provided. Defaults to None.
        gs (np.ndarray | jax.Array | None, optional): Existing history of gradients. Must be provided if `xxs` and `ys` 
            are provided. Defaults to None.
        num_iter_until_recompile (int, optional): The number of optimization iterations, where jax does not recompile functions.
            Recompilation makes the execution faster, but also takes a lot of time. Therefore, the number of recompilations is a
            hyperparameter that can be tuned for speed (depending on hardware, other parameters and JAX version).
            Defaults to 50.
        ensemble_size (int, optional): The number of individual MLP models to train within
            the ensemble surrogate. Defaults to 20.
        training_plot_directory (Path | None, optional): Directory to save plots related
            to model training (losses, etc.). Defaults to None.
        surrogate_plot_directory (Path | None, optional): Directory to save visualizations
            of the surrogate landscape. This is only available if the dimensionality of the function is 1. 
            Defaults to None.
        num_acq_optim_samples (int, optional): The number of samples used to optimize the
            acquisition function during each step. Defaults to 100.
        custom_ei_config (EIConfig | None, optional): Custom configuration for the Expected
            Improvement acquisition function. Defaults to None.
        custom_optim_config (OptimConfig | None, optional): Custom configuration for the
            optimizer (e.g., learning rate, steps). Defaults to None.
        custom_base_model_config (MLPModelConfig | None, optional): Custom architecture
            configuration for the individual MLP models. Defaults to None.
        num_embedding_channels (int, optional): The number of embedding channels used in
            the model input layer. Defaults to 1.
        non_diff_params (np.ndarray | None, optional): A boolean mask of shape `(D,)` indicating 
            which parameters are non-differentiable (True) or differentiable (False). 
            Defaults to None (all assumed differentiable). Note that the function fn still needs to return an array of
            shape (D,) for the gradients, but the values for non-diff. parameters can be arbitrary.
        num_acq_optim_runs (int, optional): The number of indepent acquisition function optimization runs that are
            performed. Since optimizing the acq. fn. is a difficult problem, restarts can increase sample quality 
            greatly. Defaults to 5.
        num_initial_acq_samples (int, optional): Number of random samples, where the maximum is selected as startpoint
            for optimizing the acquisition function. Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the full history of:
            - `xs`: Input parameters (shape `(N, D)`).
            - `ys`: Objective values (shape `(N,)`).
            - `gs`: Gradients/Auxiliary outputs (shape `(N, D)`).
            Here D is the number of dimensions and N the sample count.

    """
    assert num_bonni_iterations > 0
    assert bounds.ndim == 2 and bounds.shape[1] == 2, "bounds needs to be array of shape (degree_of_freedom, 2)"
    num_actions = bounds.shape[0]
    
    if save_path is not None and isinstance(save_path, str):
        save_path = Path(save_path)
        
    if seed is None:
        # Generate a random integer to initialize the JAX key
        # ensuring different results for every run where seed is None.
        seed = np.random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
    elif isinstance(seed, int):
        key = jax.random.PRNGKey(seed)
    elif isinstance(seed, jax.Array):
        key = seed
    else:
        raise Exception(f"Invalid seed: {seed}")
    
    # Validate non_diff_params
    if non_diff_params is not None:
        non_diff_params = np.asarray(non_diff_params, dtype=bool)
        assert non_diff_params.ndim == 1 and non_diff_params.shape[0] == num_actions, \
            f"non_diff_params must be a 1D boolean array of shape ({num_actions},), got {non_diff_params.shape}"
    else:
        non_diff_params = np.zeros(shape=(num_actions,), dtype=bool)

    # custom input configs
    if custom_ei_config is None:
        ei_cfg = EIConfig()
    else:
        ei_cfg = custom_ei_config
    
    if custom_optim_config is None:
        optim_cfg = OptimConfig(
            total_steps=int(1e3),
            warmup_steps=30,
        )
    else:
        optim_cfg = custom_optim_config
        
    if custom_base_model_config is None:
        base_model_cfg = MLPModelConfig(
            num_layer=4,
            out_channels=1,
            hidden_channels=256,
            norm_groups=8,
        )
    else:
        base_model_cfg = custom_base_model_config
    model_cfg = MLPEnsembleConfig(
        base_cfg=base_model_cfg,
        num_models=ensemble_size,
    )
    
    bounds = np.asarray(bounds, dtype=float)
    if xs is not None:
        xs = np.asarray(xs, dtype=float)
    if ys is not None:
        ys = np.asarray(ys, dtype=float)
    if gs is not None:
        gs = np.asarray(gs, dtype=float)
    
    
    wrapper = FunctionWrapper(
        fn=fn,
        action_bounds=bounds,
        negate=(direction != "maximize"),
        save_path=save_path,
    )
    
    # parse previous points
    x_list, y_list, g_list, num_prev_samples = [], [], [], 0
    if xs is not None or ys is not None or gs is not None:
        assert xs is not None and ys is not None and gs is not None, "if previous samples given, all xs, ys and gs need to be provided"
        assert xs.ndim == 2 and xs.shape[1] == num_actions, f"invalid shape for xs: {xs.shape}"
        num_prev_samples = xs.shape[0]
        assert ys.ndim == 1 and ys.shape[0] == num_prev_samples, f"invalid shape for ys: {ys.shape}"
        assert gs.ndim == 2 and gs.shape == xs.shape, f"invalid shape for gs: {gs.shape}"
        x_list = [x for x in xs]
        y_list = [y for y in ys]
        g_list = [g for g in gs]
    
    # sample random initial points
    if num_random_samples is not None:
        assert num_random_samples >= 0, f"invalid random sample count: {num_random_samples}"
        key, subkey = jax.random.split(key)
        random_actions = jax.random.uniform(subkey, shape=(num_random_samples, num_actions,))
        random_actions_np = np.asarray(random_actions, dtype=float)
        action_ranges = bounds[:, 1] - bounds[:, 0]
        random_xs = random_actions_np * action_ranges + bounds[:, 0]
        for idx in range(num_random_samples):
            cur_y, cur_g = wrapper(random_xs[idx])
            x_list.append(random_xs[idx])
            y_list.append(cur_y)
            g_list.append(cur_g)
    
    assert x_list, f"Need to either specify random sample count or provide previous samples, got {num_random_samples=} and {num_prev_samples=}"
    assert len(x_list) >= 2, "Bonni needs at least two samples to start optimization"
    xs = np.asarray(x_list, dtype=float)
    ys = np.asarray(y_list, dtype=float)
    gs = np.asarray(g_list, dtype=float)
    
    xs, ys, gs = bo_loop(
        fn=wrapper,
        bounds=bounds,
        key=key,
        xs=xs,
        ys=ys,
        gs=gs,
        samples_after_init=num_bonni_iterations,
        ei_cfg=ei_cfg,
        ensemble_cfg=model_cfg,
        optim_cfg=optim_cfg,
        training_plot_directory=training_plot_directory,
        surrogate_plot_directory=surrogate_plot_directory,
        num_acq_optim_samples=num_acq_optim_samples,
        num_embedding_channels=num_embedding_channels,
        num_iter_until_recompile=num_iter_until_recompile,
        non_diff_params=non_diff_params,
        num_acq_runs=num_acq_optim_runs,
        num_initial_acq_samples=num_initial_acq_samples,
    )
    
    return xs, ys, gs