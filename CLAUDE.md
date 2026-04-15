# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BONNI is a **Bayesian Optimization with Neural Network surrogates** library. It optimizes black-box objective functions by using gradient information (in addition to function values) to train MLP ensemble surrogates, then maximizes an Expected Improvement acquisition function via IPOPT.

## Commands

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
uv run python -m pytest tests --cov --cov-branch --cov-config=pyproject.toml --cov-report=xml

# Run a single test file
uv run python -m pytest tests/test_bonni.py

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uvx ty check --error-on-warning

# Build docs
uv run sphinx-build -W --keep-going docs/source/ docs/build/
```

Pre-commit hooks (ruff, type checking, YAML validation) run automatically on commit. To run manually: `uv run pre-commit run -a`.

## Architecture

### Entry Points

- `src/bonni/bonni.py` — `optimize_bonni()`: main BO loop, the primary user-facing function
- `src/bonni/ipopt.py` — `optimize_ipopt()`: standalone IPOPT wrapper for gradient-based optimization

### BO Loop (`src/bonni/bo.py`)

Core iteration: train surrogate → optimize acquisition function → query objective → repeat.

Key detail: arrays are **pre-allocated with padding** to avoid JAX recompilation on each iteration. When the buffer fills, it is doubled and JAX recompiles once.

### Neural Network Surrogate (`src/bonni/model/`)

- `ensemble.py`: `MLPEnsemble` — a set of independently initialized MLPs; ensemble variance provides uncertainty estimates
- `mlp.py`: Flax MLP layer and architecture
- `embedding.py`: `SinCosActionEmbedding` — sinusoidal positional encoding for bounded inputs (similar to transformer embeddings)
- `training.py`: `full_regression_training_bnn()` — trains with both value **and gradient** supervision; this is BONNI's key differentiator from standard BO
- `optim.py`: optimizer/scheduler configuration (Optax)

### Acquisition Function (`src/bonni/acquisition/`)

- `ei.py`: Expected Improvement with numerical stability handling (log-space computation)
- `optim.py`: `AcqFnWrapper` — wraps EI as an IPOPT-compatible objective to find the next query point

### Supporting Modules

- `function.py`: `FunctionWrapper` — wraps user functions, counts evaluations, handles NPZ save/load of `(xs, ys, gs)` history
- `synthetic.py`: synthetic test functions for benchmarking
- `constants.py`: shared type aliases
- `visualization/`: landscape and training metric plots

### Data Flow

```
User function (returns value + gradient)
    → FunctionWrapper (tracks evaluations)
    → BO loop:
        1. Train MLPEnsemble on (xs, ys, gs)
        2. AcqFnWrapper optimizes EI via IPOPT → next point x*
        3. Evaluate function at x*
    → Returns (xs, ys, gs) history
```

### Key Conventions

- **Objective functions** must return `(float, np.ndarray)` — value and gradient
- **Bounds** shape is `(n_dims, 2)` with columns `[lower, upper]`
- **Direction**: `"minimize"` (default) or `"maximize"` — handled internally by sign flip
- **JAX/NumPy boundary**: inputs/outputs are NumPy; internal computation uses JAX arrays
- Configuration is done via dataclasses (`MLPModelConfig`, `EIConfig`, `OptimConfig`)
