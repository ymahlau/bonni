# This script shows how to optimize a distributed bragg reflector using BONNI.
# This can be used to reproduce the results from the paper
# IMPORTANT: to run this script you need to install the TMM library:
# pip install tmm

from pathlib import Path
import sys
from bonni import ActivationType, EIConfig, MLPModelConfig, OptimConfig, optimize_bonni
from bonni.misc import change_to_timestamped_dir
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass

# Import the standard tmm library
from tmm import coh_tmm

# --- Constants ---
N_AIR = 1.0
N_SIO2 = 1.46
N_TIO2 = 2.5


@dataclass(frozen=True)
class TMMEnvConfig:
    compute_gradients: bool = True
    scale_factor: float = 100e-9

    @property
    def num_layer(self) -> int:
        return 10

    def get_material_list(self):
        mat_list = ["Air"]
        # Alternating layers
        for _ in range(self.num_layer // 2):
            mat_list.append("TiO2")
            mat_list.append("SiO2")
        mat_list.append("Air")
        return mat_list


class TMMEnv:
    def __init__(self, cfg: TMMEnvConfig):
        self.cfg = cfg
        # Cache material names to avoid regenerating them every call
        self._material_names = self.cfg.get_material_list()

    @property
    def name(self) -> str:
        return "TMM_Standard"

    @property
    def num_actions(self) -> int:
        return self.cfg.num_layer

    @property
    def action_bounds(self) -> jax.Array:
        lists = []
        for mat in self.cfg.get_material_list():
            if mat == "Air":
                continue
            if mat == "TiO2":
                lists.append([26.6e-9, 239.4e-9])
            if mat == "SiO2":
                lists.append([45.5e-9, 409.9e-9])

        bounds = jnp.asarray(lists) / self.cfg.scale_factor
        return bounds

    # --- Helper: Refractive Index Lookup ---
    def _get_n_list(self) -> list[float]:
        """Constructs the list of refractive indices based on material names."""
        n_list = []
        for mat in self._material_names:
            if mat == "Air":
                val = N_AIR
            elif mat == "SiO2":
                val = N_SIO2
            elif mat == "TiO2":
                val = N_TIO2
            else:
                val = 1.0
            n_list.append(val)
        return n_list

    # --- New Function: Compute Transmission ---
    def compute_transmission(
        self, thicknesses: np.ndarray, num_points: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the transmission spectrum for given layer thicknesses.

        Args:
            thicknesses: NumPy array of physical layer thicknesses (in meters).
            num_points: Number of wavelength points to simulate.

        Returns:
            wavelengths: NumPy array of wavelengths.
            transmission: NumPy array of transmission values (0.0 to 1.0).
        """
        # 1. Setup Simulation Range
        wl_min, wl_max = 500e-9, 800e-9
        wavelengths = np.linspace(wl_min, wl_max, num_points)

        # 2. Construct Thickness List (tmm format: [inf, layers..., inf])
        d_list = [float("inf")] + thicknesses.tolist() + [float("inf")]

        # 3. Get Refractive Indices
        n_list = self._get_n_list()

        transmission_spectrum = []

        # 4. TMM Loop
        for lam in wavelengths:
            # Run TMM: pol='s', theta=0
            tmm_data = coh_tmm("s", n_list, d_list, 0, lam)
            transmission_spectrum.append(tmm_data["T"])

        return wavelengths, np.array(transmission_spectrum)

    def get_target(self, wavelengths: np.ndarray) -> np.ndarray:
        target = np.where(wavelengths < 620e-9, 0.0, 1.0)
        return target

    def query(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert actions.ndim == 1
        assert actions.shape[0] == self.num_actions

        # 2. Define Loss Calculation Logic
        def forward_fn(current_actions):
            # Scale actions to physical thicknesses
            thicknesses = current_actions * self.cfg.scale_factor

            wavelengths, transmission = self.compute_transmission(
                thicknesses, num_points=100
            )

            target = self.get_target(wavelengths)

            # Calculate MAE Loss
            loss = np.mean(np.abs(transmission - target))
            return -loss

        # 3. Calculate Reward (Forward Pass)
        reward = forward_fn(actions)

        # 4. Calculate Gradients (Finite Difference) if required
        grads = None
        if self.cfg.compute_gradients:
            grads = np.zeros_like(actions)
            epsilon = 1e-6

            for i in range(len(actions)):
                act_perturbed = actions.copy()
                act_perturbed[i] += epsilon
                reward_perturbed = forward_fn(act_perturbed)
                grads[i] = (reward_perturbed - reward) / epsilon

        return np.asarray(reward, dtype=float), np.asarray(grads, dtype=float)


def main(seed: int):
    change_to_timestamped_dir(
        file=Path(__file__),
        # fixed_time_folder=f"bonni_tmm_{seed}",
    )

    af_cfg = EIConfig(
        offset=1e-4,
    )
    base_model_cfg = MLPModelConfig(
        num_layer=4,
        out_channels=1,
        hidden_channels=256,
        norm_groups=8,
        activation_type=ActivationType.gelu,
        different_last_activation=ActivationType.identity,
        skip_if_possible=True,
    )
    optim_cfg = OptimConfig(
        total_steps=int(1e3),
        warmup_steps=30,
    )
    env_cfg = TMMEnvConfig()
    env = TMMEnv(env_cfg)

    optimize_bonni(
        fn=env.query,
        bounds=np.asarray(env.action_bounds, dtype=float),
        num_bonni_iterations=1000,
        num_random_samples=50,
        num_iter_until_recompile=50,
        ensemble_size=25,
        num_embedding_channels=1,
        num_acq_optim_samples=200,
        direction="maximize",
        save_path=Path.cwd(),
        seed=seed,
        custom_ei_config=af_cfg,
        custom_base_model_config=base_model_cfg,
        custom_optim_config=optim_cfg,
        num_acq_optim_runs=10,
        num_initial_acq_samples=100,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = 0
    main(idx)
