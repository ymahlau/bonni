# This script shows how to optimize a dual-layer grating coupler using BONNI.
# This can be used to reproduce the results from the paper
# IMPORTANT: to run this script you need to install the tidy3d library:
# pip install tidy3d
# Also note that running this script will lead to costs since tidy3d is a commercial library


from dataclasses import dataclass
import math
from pathlib import Path
from autograd.tracer import getval
import sys
import time

from bonni import ActivationType, EIConfig, MLPModelConfig, OptimConfig, optimize_bonni
from bonni.misc import change_to_timestamped_dir
import numpy as onp
import autograd.numpy as np
import autograd

import tidy3d as td
from tidy3d import web


@dataclass(frozen=True, kw_only=True)
class GCEnvConfig:
    compute_gradients: bool = True
    full_grating_control: bool = True
    include_first_si_gap: bool = True
    include_first_sin_gap: bool = True

    num_elements: int = 15

    box_thickness: float = 2.0
    si_thickness: float = 0.09
    sin_thickness: float = 0.4
    bandwidth: float = 0.1
    freq_points: float = 101
    beam_offset_x: float = 5.0
    beam_height: float = 2.0
    beam_mfd: float = 9.2
    beam_angle_deg: float = 10

    inf: float = 1000
    buffer_left: float = 3.0
    buffer_right: float = 3.0
    buffer_bot: float = 2.0
    buffer_top: float = 0.5

    substrate_index: float = 3.47
    box_index: float = 1.44
    si_index: float = 3.47
    sin_index: float = 2.0

    min_width_si: float = 0.1
    min_gap_si: float = 0.1
    min_width_sin: float = 0.2
    min_gap_sin: float = 0.3
    max_width_si: float = 1.0
    max_gap_si: float = 1.0
    max_width_sin: float = 1.0
    max_gap_sin: float = 1.0

    default_first_gap_si: float = -0.7
    spacer_thickness: float = 0.3

    center_wavelength: float = 1.55
    min_steps_per_wvl: float = 20
    run_time: float = 1e-12

    num_retry_after_failure: int = 10

    @property
    def default_first_gap_sin(self) -> float:
        return 1.5 * self.min_gap_sin


class GCEnv:
    def __init__(
        self,
        cfg: GCEnvConfig,
    ):
        self.cfg = cfg
        self.substrate = td.Medium(permittivity=self.cfg.substrate_index**2)
        self.box = td.Medium(permittivity=self.cfg.box_index**2)
        self.si = td.Medium(permittivity=self.cfg.si_index**2)
        self.sin = td.Medium(permittivity=self.cfg.sin_index**2)

    @property
    def name(self) -> str:
        return "2-Layer Grating Coupler"

    @property
    def num_actions(self) -> int:
        num_actions = 4 * self.cfg.num_elements if self.cfg.full_grating_control else 4
        if self.cfg.include_first_si_gap:
            num_actions += 1
        if self.cfg.include_first_sin_gap:
            num_actions += 1
        return num_actions

    @property
    def num_base_actions(self) -> int:
        return self.cfg.num_elements * 4 if self.cfg.full_grating_control else 4

    @property
    def action_bounds(self):
        # order is base_actions, si_gap, sin_gap
        # base actions is ordered by widths_si, gaps_si, widths_sin, gaps_sin
        bound_list = [
            [self.cfg.min_width_si, self.cfg.max_width_si],
            [self.cfg.min_gap_si, self.cfg.max_gap_si],
            [self.cfg.min_width_sin, self.cfg.max_width_sin],
            [self.cfg.min_gap_sin, self.cfg.max_gap_sin],
        ]
        if self.cfg.full_grating_control:
            bound_list = (
                [bound_list[0] for _ in range(self.cfg.num_elements)]
                + [bound_list[1] for _ in range(self.cfg.num_elements)]
                + [bound_list[2] for _ in range(self.cfg.num_elements)]
                + [bound_list[3] for _ in range(self.cfg.num_elements)]
            )
        if self.cfg.include_first_si_gap:
            bound_list += [
                [
                    self.cfg.default_first_gap_si - 0.2,
                    self.cfg.default_first_gap_si + 0.2,
                ]
            ]
        if self.cfg.include_first_sin_gap:
            bound_list += [
                [
                    self.cfg.default_first_gap_sin - 0.4,
                    self.cfg.default_first_gap_sin + 0.2,
                ]
            ]
        return onp.asarray(bound_list, dtype=onp.float64)

    def get_simulation(self, actions_np, include_field_monitor: bool = False):
        # base actions
        n = self.cfg.num_elements
        if self.cfg.full_grating_control:
            widths_si = actions_np[:n]
            gaps_si = actions_np[n : 2 * n]
            widths_sin = actions_np[2 * n : 3 * n]
            gaps_sin = actions_np[3 * n : 4 * n]
        else:
            widths_si = np.ones(shape=(self.cfg.num_elements,)) * actions_np[0]  # type: ignore
            gaps_si = np.ones(shape=(self.cfg.num_elements,)) * actions_np[1]  # type: ignore
            widths_sin = np.ones(shape=(self.cfg.num_elements,)) * actions_np[2]  # type: ignore
            gaps_sin = np.ones(shape=(self.cfg.num_elements,)) * actions_np[3]  # type: ignore
        # first si gap
        first_gap_si = None
        idx = self.num_base_actions
        if self.cfg.include_first_si_gap:
            first_gap_si = actions_np[idx]
            idx += 1
        # first sin gap
        first_gap_sin = None
        if self.cfg.include_first_sin_gap:
            first_gap_sin = actions_np[idx]
            idx += 1

        sim = self.make_simulation(
            widths_si=widths_si,
            gaps_si=gaps_si,
            widths_sin=widths_sin,
            gaps_sin=gaps_sin,
            first_gap_si=first_gap_si,
            first_gap_sin=first_gap_sin,
            include_field_monitor=include_field_monitor,
        )
        return sim

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
                task_name="gc",
                path="gc_results.hdf5",  # ty:ignore[invalid-argument-type]
            )

            power_da = self.get_mode_monitor_power(sim_data)
            power = np.squeeze(power_da.data)  # type: ignore
            center_idx = math.floor(len(power) / 2)
            center_power = power[center_idx].flatten()
            return center_power

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

    def widths_gaps_to_centers(self, widths, gaps, *, first_gap):
        widths = np.array(widths)
        gaps = np.array(gaps)
        n = widths.size

        gaps_interior = gaps[: n - 1] if n > 1 else gaps[:0]
        if n == 0:
            return widths[:0], widths

        combined = widths[:-1] + gaps_interior
        cumulative = np.cumsum(combined)  # type: ignore
        prefix_offset = np.zeros(1, dtype=widths.dtype)  # type: ignore
        prefix = (
            np.concatenate((prefix_offset, cumulative))
            if cumulative.size
            else prefix_offset
        )
        centers = first_gap + prefix + widths / 2
        return centers, widths

    def make_grating_structures(
        self,
        widths_si,
        gaps_si,
        widths_sin,
        gaps_sin,
        *,
        first_gap_si,
        first_gap_sin,
        box_thickness,
        si_thickness,
        spacer_thickness,
        sin_thickness,
    ):
        """Return tidy3d structures for the dual-layer grating.

        metadata dict contains domain bounds and grating start for downstream use.
        """
        c_si, w_si = self.widths_gaps_to_centers(
            widths_si, gaps_si, first_gap=first_gap_si
        )
        c_sin, w_sin = self.widths_gaps_to_centers(
            widths_sin, gaps_sin, first_gap=first_gap_sin
        )

        structures = []
        substrate_geom = td.Box.from_bounds(
            (-self.cfg.inf, -self.cfg.inf, -self.cfg.inf),
            (self.cfg.inf, self.cfg.inf, 0),
        )
        structures.append(
            td.Structure(
                geometry=substrate_geom,
                medium=self.substrate,
                name="substrate",
            )
        )

        si_teeth = [
            td.Box(
                center=(
                    center,
                    0,
                    substrate_geom.bounds[1][2] + box_thickness + si_thickness / 2,
                ),
                size=(width, self.cfg.inf, si_thickness),
            )
            for center, width in zip(c_si, w_si)
        ]
        structures.append(
            td.Structure(
                geometry=td.GeometryGroup(geometries=si_teeth),  # type: ignore
                medium=self.si,
                name="si_teeth",
            )
        )

        sin_waveguide_geom = td.Box.from_bounds(
            (-self.cfg.inf, -self.cfg.inf, si_teeth[0].bounds[1][2] + spacer_thickness),
            (
                0,
                self.cfg.inf,
                si_teeth[0].bounds[1][2] + spacer_thickness + sin_thickness,
            ),
        )
        structures.append(
            td.Structure(
                geometry=sin_waveguide_geom,
                medium=self.sin,
                name="sin_waveguide",
            )
        )

        sin_teeth = [
            td.Box(
                center=(center, 0, sin_waveguide_geom.center[2]),  # type: ignore
                size=(width, self.cfg.inf, sin_thickness),
            )
            for center, width in zip(c_sin, w_sin)
        ]
        structures.append(
            td.Structure(
                geometry=td.GeometryGroup(geometries=sin_teeth),  # type: ignore
                medium=self.sin,
                name="sin_teeth",
            )
        )

        return structures, {
            "c_sin": sin_waveguide_geom.center,
            "x_gc": max(sin_teeth[-1].bounds[1][0], si_teeth[-1].bounds[1][0]),
        }

    def make_simulation(
        self,
        widths_si,
        gaps_si,
        widths_sin,
        gaps_sin,
        first_gap_si=None,
        first_gap_sin=None,
        *,
        include_field_monitor=False,
    ):
        if first_gap_si is None:
            first_gap_si = self.cfg.default_first_gap_si
        if first_gap_sin is None:
            first_gap_sin = self.cfg.default_first_gap_sin
        structures, info = self.make_grating_structures(
            widths_si,
            gaps_si,
            widths_sin,
            gaps_sin,
            first_gap_si=first_gap_si,
            first_gap_sin=first_gap_sin,
            box_thickness=self.cfg.box_thickness,
            si_thickness=self.cfg.si_thickness,
            spacer_thickness=self.cfg.spacer_thickness,
            sin_thickness=self.cfg.sin_thickness,
        )

        freq0 = td.C_0 / self.cfg.center_wavelength
        freqs = td.C_0 / np.linspace(  # type: ignore
            self.cfg.center_wavelength - self.cfg.bandwidth / 2,
            self.cfg.center_wavelength + self.cfg.bandwidth / 2,
            self.cfg.freq_points,
        )

        source_z = info["c_sin"][2] + self.cfg.sin_thickness / 2 + self.cfg.beam_height
        source = td.GaussianBeam(
            center=(self.cfg.beam_offset_x, 0, source_z),
            size=(self.cfg.inf, self.cfg.inf, 0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
            pol_angle=np.pi / 2,  # type: ignore
            angle_theta=np.deg2rad(self.cfg.beam_angle_deg),  # type: ignore
            direction="-",
            waist_radius=self.cfg.beam_mfd / 2,
            name="input_beam",
        )

        monitors = [
            td.ModeMonitor(
                center=(-self.cfg.buffer_left + 0.5, 0, getval(info["c_sin"][2])),
                size=(0, self.cfg.inf, 3),
                freqs=freqs,
                mode_spec=td.ModeSpec(num_modes=1),
                name="mode_monitor",
            )
        ]

        if include_field_monitor:
            monitors.append(
                td.FieldMonitor(
                    center=(0, 0, 0),
                    size=(self.cfg.inf, 0, self.cfg.inf),
                    freqs=freq0,
                    fields=("Ey",),
                    name="field_monitor",
                )
            )

        x_min = getval(-self.cfg.buffer_left)
        x_max = getval(info["x_gc"] + self.cfg.buffer_right)
        z_min = getval(-self.cfg.buffer_bot)
        z_max = getval(source_z + self.cfg.buffer_top)

        simulation = td.Simulation(
            center=((x_min + x_max) / 2, 0, (z_min + z_max) / 2),
            size=(x_max - x_min, 0, z_max - z_min),
            structures=structures,
            sources=(source,),
            monitors=tuple(monitors),
            medium=self.box,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.periodic(),
                z=td.Boundary.pml(),
            ),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=self.cfg.min_steps_per_wvl),
            run_time=self.cfg.run_time,
        )
        return simulation

    def get_mode_monitor_power(
        self,
        sim_data,
        *,
        mode_index=0,
        direction="-",
        monitor_name="mode_monitor",
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

    env_cfg = GCEnvConfig()
    env = GCEnv(cfg=env_cfg)

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
