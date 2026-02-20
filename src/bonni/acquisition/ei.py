from dataclasses import dataclass
from typing import Literal
import jax
import numpy as np
import jax.numpy as jnp
import scipy
import math

from bonni.model.embedding import SinCosActionEmbedding
from bonni.model.ensemble import MLPEnsemble

_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))

# -------------------------------------------------------------------------
# 1. Pure NumPy Logic (Univariate: z -> log_ei)
# -------------------------------------------------------------------------


def _standard_logei_numpy(z_input):
    """
    Calculates log(EI_standard(z)).
    Input 'z' is handled as a NumPy array.
    """
    z = np.asarray(z_input, dtype=float)

    # Ensure operation on at least 0-d array (handles float scalars)
    # This prevents 'float object does not support item assignment'
    if z.ndim == 0:
        z = z[None]
        was_scalar = True
    else:
        was_scalar = False

    # Case 1: Standard calculation (z >= -25)
    # Using small epsilon to prevent -inf in log during temporary calculation
    z_half = 0.5 * z
    term1 = z_half * scipy.special.erfc(-_SQRT_HALF * z)
    term2 = np.exp(-z_half * z) * _INV_SQRT_2PI
    out = np.log(np.maximum(term1 + term2, 1e-100))

    # Case 2: Asymptotic expansion (z < -25)
    mask_small = z < -25
    if np.any(mask_small):
        z_small = z[mask_small]
        # val = (
        #     -0.5 * (z_small ** 2)
        #     - _LOG_SQRT_2PI
        #     + np.log(1 + _SQRT_HALF_PI * z_small * scipy.special.erfcx(-_SQRT_HALF * z_small))
        # )
        val = -0.5 * (z_small**2) - _LOG_SQRT_2PI - 2.0 * np.log(np.abs(z_small))
        out[mask_small] = val

    out = out.astype(z_input.dtype)
    if was_scalar:
        return out[0]
    return out


def _standard_logei_grad_numpy(z, g, y):
    """
    Calculates gradient of standard_logei w.r.t z.
    d(LogEI)/dz = exp(log_cdf(z) - log_ei(z))
    """
    z = np.asarray(z, dtype=float)
    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float)  # y is the precomputed log_ei(z)

    # log(CDF(z))
    log_phi_cdf = scipy.special.log_ndtr(z)

    # d(LogEI)/dz = exp(log_Phi - log_EI)
    # Note: y is log_EI
    grad_z = g * np.exp(log_phi_cdf - y)

    return grad_z.astype(z.dtype)


# -------------------------------------------------------------------------
# 2. JAX Primitives
# -------------------------------------------------------------------------


@jax.custom_vjp
def standard_logei(z):
    """
    JAX primitive for the standard LogEI function.
    Accepts JAX array 'z', returns JAX array.
    """
    # Output shape is same as input z
    result_shape = jax.ShapeDtypeStruct(z.shape, z.dtype)

    return jax.pure_callback(
        _standard_logei_numpy,
        result_shape,
        z,
        vmap_method="expand_dims",  # Handle batches efficiently
    )


def _standard_logei_fwd(z):
    y = standard_logei(z)
    return y, (z, y)


def _standard_logei_bwd(res, g):
    z, y = res
    z_grad_shape = jax.ShapeDtypeStruct(z.shape, z.dtype)

    d_z = jax.pure_callback(
        _standard_logei_grad_numpy, z_grad_shape, z, g, y, vmap_method="expand_dims"
    )
    return (d_z,)


standard_logei.defvjp(_standard_logei_fwd, _standard_logei_bwd)


# -------------------------------------------------------------------------
# 3. Main Function & Class
# -------------------------------------------------------------------------


def log_expected_improvement(mean, std, f0):
    """
    Composes the LogEI calculation using JAX ops and the custom primitive.

    LogEI(mu, sigma, f0) = log(sigma * EI_std(z))
                         = log(sigma) + LogEI_std(z)

    where z = (mu - f0) / sigma
    """
    # JAX handles broadcasting of mean, std, and f0 here automatically.
    # This prevents the shape mismatch errors inside the callback.
    z = (mean - f0) / std

    # Call the custom primitive for the hard part
    log_ei_z = standard_logei(z)

    # Combine
    return log_ei_z + jnp.log(std)


@dataclass(kw_only=True, frozen=True)
class EIConfig:
    """
    Configuration for the Expected Improvement acquisition function. Specifically, this can be controlled to impose
    penalties on the evaluation of new samples. For example with penalty_mode='bounds', sampling near boundaries is
    penalized, or with penalty_mode='distance' sampling far from previous samples is penalized.

    Attributes:
        offset (float): Offset for increasing the exploration during optimization. by default this is a small positive
            value. Defaults to 1e-4.
        stop_penalty_after (int | None): Number of optimization iterations, after which no more penalty is applied.
            We recommend setting this to half the number of total iterations. Defaults to None.
        penalty_mode (Literal['none', 'bounds', 'distance']): Mode for penalizing different sampling behavior.
            With 'none', no penalty is applied.
            With 'bounds' sampling near boundary is penalized.
            With 'distance', samples far from previous samples are penalized.
            Defaults to 'none'.
        distance_threshold (float): Penalty threshold for the distance mode. This has to be value in the range [0, 1].
            Defaults to 0.3.
        penalty_weight (float): Scale of the penalty applied. This should be roughly equivalent to the
            range between the minimum and maximum possible value of the objective function. Defaults to 1.0.
        bounds_threshold (float): Penalty threshold for the bounds mode. This has to be a value in the range [0, 0.5].
            Defaults to 0.25.
    """

    offset: float = 1e-4
    penalty_mode: Literal["none", "bounds", "distance"] = "none"
    stop_penalty_after: int | None = None
    neighbor_threshold: float = 0.3
    value_factor: float = 1.0
    boundary_penalty_start: float = 0.25


class ExpectedImprovement:
    def __init__(
        self,
        cfg: EIConfig,
    ):
        self.cfg = cfg

    def calculate_offset(
        self,
        x_test: jax.Array,
        bounds: jax.Array,
        xs: jax.Array,
        sample_mask: jax.Array,
    ) -> jax.Array:
        if self.cfg.penalty_mode == "none":
            return jnp.asarray(self.cfg.offset)

        assert x_test.ndim == 1
        assert xs.ndim == 2
        assert xs.shape[1] == x_test.shape[0]

        # --- FIX 1: Check actual sample count, not buffer size ---
        # xs.shape[0] includes padding. We sum the mask to get real count.
        num_valid_samples = jnp.sum(sample_mask)

        # Note: If you want to return early inside a JIT function, you usually can't.
        # Instead, we will compute the penalty and mask it out at the end if the condition is met.
        if (
            self.cfg.stop_penalty_after is not None
            and xs.shape[0] > self.cfg.stop_penalty_after
        ):
            return jnp.asarray(self.cfg.offset)

        if self.cfg.penalty_mode == "distance":
            # 1. Calculate Distances to ALL points (including padding)
            distances = jnp.sqrt(jnp.sum(jnp.square(x_test[None, :] - xs), axis=-1))

            # --- FIX 2: Mask out padded samples ---
            # Set distance to invalid samples to Infinity so jnp.min ignores them.
            # (Otherwise, padded zeros look like a neighbor at the origin).
            distances = jnp.where(sample_mask, distances, jnp.inf)

            # Find distance to closest VALID neighbor
            point_distance = jnp.min(distances)

            # Handle edge case: if NO valid samples exist (all mask False), point_distance is inf.
            # We clamp it or handle it.
            # If inf (first iter), logic below (1 - edge/inf) -> 1.0 -> Max Penalty.
            # This makes sense (don't trust anything yet), or you might want 0 penalty.

            ld, ud = x_test - bounds[:, 0], bounds[:, 1] - x_test
            min_d = jnp.minimum(ld, ud)
            edge_distance = jnp.min(min_d)
            edge_distance = jnp.where(edge_distance < 1e-6, 1e-6, edge_distance)

            threshold = (point_distance + edge_distance) * self.cfg.neighbor_threshold
            threshold = jnp.where(threshold < 1e-6, 1e-6, threshold)

            penalty_factor = jnp.maximum(1 - edge_distance / threshold, 0)
            result = penalty_factor * self.cfg.value_factor

            # Apply the "stop_penalty_after" logic via masking
            if self.cfg.stop_penalty_after is not None:
                is_active = num_valid_samples <= self.cfg.stop_penalty_after
                # If active, return result. If not, return default offset.
                return jnp.where(is_active, result, self.cfg.offset)

            return result

        # bounds penalty (Mask irrelevant here as it only uses x_test and bounds)
        half_value_position = self.cfg.boundary_penalty_start / 2
        normalized_x = (x_test - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        centered_distance = jnp.abs(normalized_x - 0.5)
        outer_start, outer_end = 0.5, 0.5 - half_value_position
        is_outer_range = centered_distance > outer_end
        outer_range_position = (centered_distance - outer_start) / (
            outer_end - outer_start
        )
        mid_start, mid_end = (
            (0.5 - half_value_position),
            (0.5 - self.cfg.boundary_penalty_start),
        )
        is_mid_range = (mid_start >= centered_distance) & (centered_distance > mid_end)
        mid_range_position = (centered_distance - mid_start) / (mid_end - mid_start)

        value = self.cfg.value_factor
        cur_offset = jnp.ones_like(x_test) * self.cfg.offset
        cur_offset = jnp.where(
            is_outer_range,
            (1 - outer_range_position) * 0.5 * value + 0.5 * value,
            cur_offset,
        )
        cur_offset = jnp.where(
            is_mid_range,
            (1 - mid_range_position) * (0.5 * value - self.cfg.offset)
            + self.cfg.offset,
            cur_offset,
        )

        mean_offset = jnp.mean(cur_offset)

        # Apply Stop Penalty logic for bounds mode too
        if self.cfg.stop_penalty_after is not None:
            is_active = num_valid_samples <= self.cfg.stop_penalty_after
            return jnp.where(is_active, mean_offset, self.cfg.offset)

        return mean_offset

    def __call__(
        self,
        x_test: jax.Array,
        xs: jax.Array,
        ys: jax.Array,
        bounds: jax.Array,
        params,
        model: MLPEnsemble,
        embedding: SinCosActionEmbedding,
        sample_mask: jax.Array,
    ) -> jax.Array:
        assert x_test.ndim == 1
        assert x_test.shape[0] == bounds.shape[0]

        # 1. Model Forward
        obs_test = embedding(x_test, bounds)
        pred = model.apply(params, obs_test)
        assert pred.ndim == 1 and pred.shape[0] == model.cfg.num_models

        # 2. Robust Statistics Calculation
        # We must ignore the padded values in ys to get the real Mean/Std
        y_safe = jnp.nan_to_num(ys, nan=0.0)

        # Use the mask to calculate stats of only valid data
        y_mean = jnp.mean(y_safe, where=sample_mask)
        y_std = jnp.std(y_safe, where=sample_mask)

        # Safety: If we have 0 or 1 samples, std might be 0 or nan.
        # Clamp to 1.0 to avoid division/multiplication issues.
        y_std = jnp.where(y_std < 1e-8, 1.0, y_std)

        # Un-normalize the model predictions
        mean_pred = jnp.mean(pred)
        std_pred = jnp.std(pred)

        # Apply transformation
        mean = mean_pred * y_std + y_mean
        std = std_pred * y_std

        # 3. Calculate Offset
        cur_offset = self.calculate_offset(x_test, bounds, xs, sample_mask=sample_mask)

        # 4. Calculate Y_Max
        # We mask invalid values with -Infinity so they are never chosen as Max (Assuming maximization)
        y_masked_for_max = jnp.where(sample_mask, y_safe, -jnp.inf)
        ymax = jnp.max(y_masked_for_max)

        # Add offset (penalty)
        ymax = ymax + cur_offset

        result = log_expected_improvement(mean, std, ymax)

        return result
