import jax
import jax.numpy as jnp
import numpy as np


class StyblinskiTangFn:
    def __init__(
        self,
        d: int,
    ):
        self.d = d
        tmp_input = jnp.ones(shape=(self.d,)) * 5  # determine max abs value
        self.factor = -1 / self._f(tmp_input, 1.0)

    @staticmethod
    def _f(x, factor):
        """
        Styblinski-Tang function for a d-dimensional input vector x.

        f(x) = (1/2) * sum(x_i^4 - 16*x_i^2 + 5*x_i)

        Global minimum: f(x*) = -39.16599*d at x* = (-2.903534,...,-2.903534)
        Domain: x_i ∈ [-5, 5]
        """
        assert x.ndim == 1
        d = x.size
        # Compute sum of (x_i^4 - 16*x_i^2 + 5*x_i)
        # we use mean here
        term = jnp.sum(x**4 - 16.0 * x**2 + 5.0 * x)

        # Complete Styblinski-Tang function: (1/2) * sum(...)
        result = 0.5 * term
        result = result + 39.16599 * d  # make f(x*) = 0

        return -result * factor

    def __call__(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert x.ndim == 1
        assert x.shape[0] == self.d
        x_jax = jnp.asarray(x)

        # Compute the Styblinski-Tang function value
        y = self._f(x_jax, self.factor)
        y = np.asarray(y, float)

        g = jax.grad(self._f)(x_jax, self.factor)
        g = np.asarray(g, float)
        return y, g

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray([[-5.0, 5.0] for _ in range(self.d)])
