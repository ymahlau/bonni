# This script shows how to optimize a broadband grating coupler using BONNI.
# This can be used to reproduce the results from the paper
# IMPORTANT: to run this script you need to install the tidy3d library:
# pip install tidy3d
# Also note that running this script will lead to costs since tidy3d is a commercial library

from dataclasses import dataclass
from pathlib import Path
import sys
import time

from bonni import ActivationType, EIConfig, MLPModelConfig, OptimConfig, optimize_bonni
from bonni.misc import change_to_timestamped_dir
import numpy as onp
import autograd.numpy as np
import autograd

import tidy3d as td
from tidy3d import web


from autograd.numpy.linalg import solve as alg_solve  # type: ignore


class DifferentiableSpline:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        self.h = (x[-1] - x[0]) / (self.n - 1)

        # Solve for derivatives (D) at each point
        self.D = self._solve_derivatives(self.y, self.h)

    def _solve_derivatives(self, y, h):
        """
        Solves the tridiagonal system for derivatives.
        Refactored to ensure matrix shapes remain (N, N).
        """
        n = y.shape[0]

        # 1. Main Diagonal: [1, 4, 4, ..., 4, 1]
        # We construct it piecewise to avoid item assignment
        mid_ones = 4.0 * np.ones(n - 2)  # type: ignore
        main_diag = np.concatenate([np.array([1.0]), mid_ones, np.array([1.0])])

        # 2. Upper Diagonal (k=1): [0, 1, 1, ..., 1]
        # The first element is 0 to decouple the first boundary equation
        up_ones = np.ones(n - 2)  # type: ignore
        up_diag = np.concatenate([np.array([0.0]), up_ones])

        # 3. Lower Diagonal (k=-1): [1, 1, ..., 1, 0]
        # The last element is 0 to decouple the last boundary equation
        lo_ones = np.ones(n - 2)  # type: ignore
        lo_diag = np.concatenate([lo_ones, np.array([0.0])])

        # 4. Assemble Matrix A (All parts are now compatible (N, N) shapes)
        A = np.diag(main_diag) + np.diag(up_diag, k=1) + np.diag(lo_diag, k=-1)  # type: ignore

        # 5. Construct RHS
        # rhs_i = 3/h * (y_{i+1} - y_{i-1})
        dy_internal = 3.0 / h * (y[2:] - y[:-2])
        rhs = np.concatenate([np.array([0.0]), dy_internal, np.array([0.0])])

        # 6. Solve
        return alg_solve(A, rhs)  # tpye: ignore

    def integrate_intervals(self, lb, ub):
        idx = np.searchsorted(self.x, lb, side="right") - 1  # type: ignore
        idx = np.clip(idx, 0, self.n - 2)  # type: ignore

        y_i = self.y[idx]
        y_ip1 = self.y[idx + 1]
        D_i = self.D[idx]
        D_ip1 = self.D[idx + 1]
        h = self.h

        a = y_i
        b = D_i
        c = 3 * (y_ip1 - y_i) / h**2 - (2 * D_i + D_ip1) / h
        d = 2 * (y_i - y_ip1) / h**3 + (D_i + D_ip1) / h**2

        x_i = self.x[0] + idx * h
        start = lb - x_i
        end = ub - x_i

        def integral_poly(t):
            return a * t + (b / 2) * t**2 + (c / 3) * t**3 + (d / 4) * t**4

        return integral_poly(end) - integral_poly(start)


@dataclass(frozen=True)
class TaperEnvConfig:
    compute_gradients: bool = True
    num_elements: int = 30

    wavelength: float = 1.25
    wavelength_range: float = 0.25
    num_freqs: int = 50
    wg_width_small: float = 0.45
    wg_width_wide: float = 4.5
    wg_length: float = 2.0
    thick: float = 0.22
    taper_length: float = 10.0
    min_steps_per_wavelength: int = 30
    pml_spacing: float = 2.0
    permittivity_material: float = 4.0
    permittivity_background: float = 2.25
    pixel_size: float = 0.02

    add_field_monitor: bool = False
    max_deviation: float = 0.1
    overscale_size: float = 0.5

    use_symmetry: bool = True
    num_retry_after_failure: int = 10

    @property
    def full_design_length(self) -> float:
        return 2 * self.wg_length + self.taper_length

    @property
    def freq0(self) -> float:
        return td.C_0 / self.wavelength

    @property
    def wavelength_min(self) -> float:
        return self.wavelength - self.wavelength_range

    @property
    def wavelength_max(self) -> float:
        return self.wavelength + self.wavelength_range

    @property
    def min_freq(self) -> float:
        return td.C_0 / self.wavelength_min

    @property
    def max_freq(self) -> float:
        return td.C_0 / self.wavelength_max

    @property
    def freqs_range(self) -> list[float]:
        return [
            td.C_0 / wl
            for wl in onp.linspace(
                self.wavelength_min, self.wavelength_max, self.num_freqs
            )
        ]

    @property
    def taper_size_y(self) -> float:
        return self.wg_width_wide + 2

    @property
    def num_pixels_x(self) -> int:
        return round(self.taper_length / self.pixel_size)

    @property
    def num_pixels_y(self) -> int:
        raw_pixels_y = round(self.taper_size_y / self.pixel_size)
        return raw_pixels_y if raw_pixels_y % 2 == 0 else raw_pixels_y + 1

    @property
    def half_num_pixels_y(self) -> int:
        return round(self.num_pixels_y / 2)

    @property
    def taper_shape(self) -> tuple[int, int]:
        return (self.num_pixels_x, self.num_pixels_y)

    @property
    def min_clip_bound(self) -> float:
        return self.wg_width_small / self.taper_size_y

    @property
    def max_clip_bound(self) -> float:
        return (self.wg_width_wide + self.overscale_size) / self.taper_size_y

    @property
    def wg_large_bound(self) -> float:
        return self.wg_width_wide / self.taper_size_y


class TaperEnv:
    def __init__(
        self,
        cfg: TaperEnvConfig,
    ):
        self.cfg = cfg

    def query(
        self,
        actions,
    ):
        assert actions.ndim == 1
        assert actions.shape[0] == self.num_actions

        def forward_fn(cur_actions):
            sim = self.get_simulation(cur_actions)

            sim_data = web.run(
                sim,
                task_name="taper",
                path=Path("taper_results.hdf5"),
            )

            power_da = self.get_mode_monitor_power(sim_data)
            power = np.min(power_da.data)  # type: ignore
            return power.flatten()

        for _ in range(self.cfg.num_retry_after_failure):
            try:
                if self.cfg.compute_gradients:
                    vg_fun = autograd.value_and_grad(forward_fn)
                    center_power, grads = vg_fun(actions)
                else:
                    center_power = forward_fn(actions)
                    grads = None
                break
            except Exception as e:
                print(e, flush=True)
                time.sleep(10)

        reward = center_power.flatten()[0]

        if self.cfg.compute_gradients:
            assert grads is not None
            assert grads.ndim == 1
            assert grads.shape[0] == self.num_actions
        else:
            assert grads is None

        assert reward.size == 1
        return reward, grads

    @property
    def name(self) -> str:
        return "Waveguide Taper"

    @property
    def num_actions(self) -> int:
        return self.cfg.num_elements

    @property
    def action_bounds(self):
        lb, ub = self.cfg.min_clip_bound, self.cfg.max_clip_bound
        center_line = onp.linspace(lb, ub, self.num_actions)
        bounds = onp.stack(
            [
                center_line - self.cfg.max_deviation,
                center_line + self.cfg.max_deviation,
            ],
            axis=-1,
        )
        clipped_bounds = onp.clip(bounds, a_min=lb, a_max=ub)

        return clipped_bounds

    def get_spline_interpolation(
        self,
        anchor_heights: np.array,  # type: ignore
        num_x: int,
    ):
        """
        Evaluates the terrain height using a differentiable CubicSpline (Autograd).
        """
        # Ensure 1D
        anchor_heights = np.atleast_1d(anchor_heights)  # type: ignore

        # 1. Define the physical x-coordinates
        pixel_size = 1.0 / num_x

        # Create evaluation points (pixels)
        lb = np.linspace(0, 1 - pixel_size, num_x)  # type: ignore
        ub = np.linspace(pixel_size, 1, num_x)  # type: ignore

        num_anchor = anchor_heights.shape[0]

        # Create anchor x-coordinates (0 to 1)
        # Corresponds to: x_anchor = jnp.linspace(0, 1, num_anchor+2)
        x_anchor = np.linspace(0, 1, num_anchor + 2)  # type: ignore

        # Concatenate bounds: [min_clip, anchors, max_clip]
        # Autograd requires np.concatenate, not manual assignment
        extrapolated_heights = np.concatenate(
            [
                np.array([self.cfg.min_clip_bound]),
                anchor_heights,
                np.array([self.cfg.wg_large_bound]),
            ]
        )

        # 2. Initialize the Differentiable Spline
        # This replaces interpax.CubicSpline
        spline = DifferentiableSpline(x_anchor, extrapolated_heights)

        # 3. Evaluate the spline integration
        # Replaces: jax.vmap(partial(spline.integrate...))
        # Autograd handles vectorization naturally via numpy broadcasting
        heights = spline.integrate_intervals(lb, ub)
        rescaled_heights = heights / pixel_size
        return rescaled_heights.flatten()

    def rasterize_spline_area(
        self,
        heights: np.ndarray,  # type: ignore
        num_pixel_y: int,
    ) -> np.ndarray:  # type: ignore
        # assumes that heights are within 0 < h < 1
        assert heights.ndim == 1
        y_pixel_half_size = 1 / (num_pixel_y * 2)
        ys = np.linspace(y_pixel_half_size, 1 - y_pixel_half_size, num_pixel_y)  # type: ignore

        # Broadcast to create the grid
        # heights: (1, grid_width)
        # ys:      (grid_height, 1)
        h_grid = heights[:, None]
        y_grid = ys[None, :]

        # Calculate the "Area" (Vertical Coverage)
        # The fraction of pixel 'y' covered by height 'h' is:
        #   If h >= y + 1: Fully covered (1.0)
        #   If h <= y:     Not covered (0.0)
        #   If y < h < y + 1: Partially covered (h - y)
        # This logic is exactly captured by subtracting and clipping.
        area_coverage = np.clip(h_grid - y_grid, -y_pixel_half_size, y_pixel_half_size)  # type: ignore
        area_rescaled = area_coverage * num_pixel_y + 0.5

        return area_rescaled

    def get_base_simulation(self):
        waveguide_small = td.Structure(
            name="waveguide_small",
            geometry=td.Box(
                center=(
                    -(self.cfg.wg_length + self.cfg.taper_length + self.cfg.pml_spacing)
                    / 2,
                    0,
                    0,
                ),
                size=(
                    self.cfg.wg_length + self.cfg.pml_spacing,
                    self.cfg.wg_width_small,
                    self.cfg.thick,
                ),
            ),
            medium=td.Medium(permittivity=self.cfg.permittivity_material),
        )

        waveguide_wide = td.Structure(
            name="waveguide_wide",
            geometry=td.Box(
                center=(
                    (self.cfg.wg_length + self.cfg.taper_length + self.cfg.pml_spacing)
                    / 2,
                    0,
                    0,
                ),
                size=(
                    self.cfg.wg_length + self.cfg.pml_spacing,
                    self.cfg.wg_width_wide,
                    self.cfg.thick,
                ),
            ),
            medium=td.Medium(permittivity=self.cfg.permittivity_material),
        )

        if (self.cfg.freq0 / 5) < 0.5 * (self.cfg.max_freq - self.cfg.min_freq):
            raise Exception("Source frequency range too small!!!!!!!")

        mode_source = td.ModeSource(
            name="mode_source",
            center=(-(self.cfg.taper_length + self.cfg.wg_length * 1.5) / 2, 0, 0),
            size=(0, td.inf, td.inf),
            source_time=td.GaussianPulse(
                freq0=self.cfg.freq0, fwidth=self.cfg.freq0 / 5
            ),
            direction="+",
        )

        mode_monitor = td.ModeMonitor(
            center=((self.cfg.taper_length + self.cfg.wg_length * 1.5) / 2, 0, 0),
            size=(0, td.inf, td.inf),
            freqs=self.cfg.freqs_range,
            mode_spec=td.ModeSpec(num_modes=1),
            name="mode",
        )

        monitors: list = [mode_monitor]
        if self.cfg.add_field_monitor:
            field_monitor = td.FieldMonitor(
                center=(0, 0, 0),
                size=(td.inf, td.inf, 0),
                freqs=[self.cfg.freq0],
                name="field",
            )
            monitors.append(field_monitor)

        symmetry = (0, -1, 1) if self.cfg.use_symmetry else (0, 0, 0)

        return td.Simulation(
            size=(
                self.cfg.full_design_length,
                self.cfg.wg_width_wide + 6,
                self.cfg.thick + 2,
            ),
            medium=td.Medium(permittivity=self.cfg.permittivity_background),
            structures=(waveguide_small, waveguide_wide),
            symmetry=symmetry,
            monitors=tuple(monitors),
            sources=(mode_source,),
            run_time=500 / self.cfg.freq0,
            boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=self.cfg.min_steps_per_wavelength,
                override_structures=[
                    td.MeshOverrideStructure(
                        geometry=self.get_taper_region_geometry(),
                        dl=(
                            self.cfg.pixel_size,
                            self.cfg.pixel_size,
                            self.cfg.pixel_size,
                        ),
                    ),
                ],
            ),
        )

    def get_taper_region_geometry(self):
        design_region_geometry = td.Box(
            center=(0, 0, 0),
            size=(self.cfg.taper_length, self.cfg.taper_size_y, self.cfg.thick),
        )
        return design_region_geometry

    def get_taper_structure(
        self,
        params: np.ndarray,  # type: ignore
    ):
        design_region_geometry = self.get_taper_region_geometry()
        scaled_params = (
            self.cfg.permittivity_material - self.cfg.permittivity_background
        ) * params[:, :, None]
        eps_data = self.cfg.permittivity_background + scaled_params
        return td.Structure.from_permittivity_array(
            eps_data=eps_data, geometry=design_region_geometry
        )

    def get_simulation(
        self,
        actions: np.ndarray,  # type: ignore
    ) -> td.Simulation:
        # map params
        clip_actions = np.clip(  # ty:ignore[unresolved-attribute]
            actions, a_min=self.action_bounds[:, 0], a_max=self.action_bounds[:, 1]
        )
        heights = self.get_spline_interpolation(
            clip_actions, num_x=self.cfg.num_pixels_x
        )
        arr = self.rasterize_spline_area(
            heights, num_pixel_y=self.cfg.half_num_pixels_y
        )
        full_arr = np.concatenate([arr[:, ::-1], arr], axis=1)

        taper_struct = self.get_taper_structure(full_arr)
        base_sim = self.get_base_simulation()
        sim = base_sim.updated_copy(structures=base_sim.structures + (taper_struct,))
        assert isinstance(sim, td.Simulation)
        return sim

    def get_mode_monitor_power(
        self,
        sim_data,
        *,
        mode_index=0,
        direction="+",
        monitor_name="mode",
        power_floor=1e-12,
    ):
        """Return |amps|^2 from a mode monitor as an xarray DataArray."""
        monitor = sim_data[monitor_name]
        amps = monitor.amps.sel(mode_index=mode_index, direction=direction)
        power = np.abs(amps) ** 2  # type: ignore
        if power_floor is not None:
            power = power.clip(min=power_floor)
        return power


def main(seed: int):
    change_to_timestamped_dir(
        file=Path(__file__),
        # fixed_time_folder=f"gc_bonni_{seed}",
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
        init_lr=1e-9,
        peak_lr=1e-3,
        final_lr=1e-5,
        use_adamw=True,
        clip_grad_norm=1,
    )

    env_cfg = TaperEnvConfig()
    env = TaperEnv(cfg=env_cfg)

    optimize_bonni(
        fn=env.query,
        bounds=env.action_bounds,
        num_bonni_iterations=100,
        num_random_samples=10,
        num_iter_until_recompile=50,
        ensemble_size=100,
        num_embedding_channels=1,
        num_acq_optim_samples=200,
        direction="maximize",
        save_path=Path.cwd(),
        seed=seed,
        custom_ei_config=af_cfg,
        custom_base_model_config=base_model_cfg,
        custom_optim_config=optim_cfg,
        num_acq_optim_runs=50,
        num_initial_acq_samples=100,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = 0
    main(idx)
