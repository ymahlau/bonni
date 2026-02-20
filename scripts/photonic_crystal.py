# This script shows how to optimize a photonic crystal waveguide transition using BONNI.
# This can be used to reproduce the results from the paper
# IMPORTANT: to run this script you need to install the tidy3d library:
# pip install tidy3d
# Also note that running this script will lead to costs since tidy3d is a commercial library


from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class CrystalTaperEnvConfig:
    compute_gradients: bool = True
    hole_radius: float = 96e-3
    num_cycles_taper: int = 10
    num_cycles_center: int = 20
    lattice_constant: float = 394e-3

    n_si: float = 3.4784
    n_sio2: float = 1.44427

    wv_center: float = 1.55
    wv_span: float = 0.1

    slab_height: float = 0.21
    wg_width: float = 1.002
    wg_length: float = 2.0

    num_retry_after_failure: int = 10
    add_field_monitor: bool = False
    use_symmetry: bool = True

    include_radius: bool = True
    min_radius: float = 40e-3
    max_radius: float = 150e-3

    @property
    def hole_diameter(self) -> float:
        return 2 * self.hole_radius

    @property
    def num_cycles_total(self) -> int:
        return 2 * self.num_cycles_taper + self.num_cycles_center

    @property
    def box_length(self) -> float:
        return (self.num_cycles_total - 1) * self.lattice_constant + self.hole_diameter

    @property
    def sim_length(self) -> float:
        return self.box_length + 2 * self.wg_length

    @property
    def freq0(self) -> float:
        return td.C_0 / self.wv_center

    @property
    def fwidth(self) -> float:
        return td.C_0 / self.wv_center * self.wv_span

    @property
    def lattice_height(self) -> float:
        return math.sqrt(self.lattice_constant**2 - (self.lattice_constant**2) / 4)

    @property
    def taper_offset(self) -> float:
        return self.wg_width / 2 - self.lattice_height


class CrystalTaperEnv:
    def __init__(
        self,
        cfg: CrystalTaperEnvConfig,
    ):
        self.cfg = cfg

    def query(
        self,
        actions,
    ):
        assert actions.ndim == 1
        assert actions.shape[0] == self.num_actions

        def forward_fn(cur_actions):
            sim = self.get_full_sim(cur_actions)

            sim_data = web.run(
                sim,
                task_name="taper",
                path=Path("taper_results.hdf5"),
            )

            power_da = self.get_mode_monitor_power(sim_data)
            power = np.squeeze(power_da.data)  # type: ignore
            return power.flatten()

        for idx in range(self.cfg.num_retry_after_failure):
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
                if idx == self.cfg.num_retry_after_failure - 1:
                    raise e

        reward = center_power.flatten()[0]

        if grads is not None:
            assert grads.ndim == 1
            assert grads.shape[0] == self.num_actions

        assert reward.size == 1
        return reward, grads

    @property
    def name(self) -> str:
        return "Photonic Crystal Waveguide Taper"

    @property
    def num_actions(self) -> int:
        return 90 if self.cfg.include_radius else 60

    @property
    def action_bounds(self):
        bound_list = [[-self.cfg.hole_radius, self.cfg.hole_radius] for _ in range(60)]
        if self.cfg.include_radius:
            radius_list = [
                [self.cfg.min_radius, self.cfg.max_radius] for _ in range(30)
            ]
            bound_list.extend(radius_list)

        return np.asarray(bound_list, dtype=float)  # type: ignore

    def get_base_sim(self) -> td.Simulation:
        mat_si = td.Medium(permittivity=self.cfg.n_si**2)
        mat_sio2 = td.Medium(permittivity=self.cfg.n_sio2**2)

        waveguide = td.Structure(
            name="waveguide",
            geometry=td.Box(
                center=(0, 0, 0),
                size=(td.inf, self.cfg.wg_width, self.cfg.slab_height),
            ),
            medium=mat_si,
        )

        box = td.Structure(
            name="box",
            geometry=td.Box(
                center=(0, 0, 0),
                size=(self.cfg.box_length, td.inf, self.cfg.slab_height),
            ),
            medium=mat_si,
        )

        pulse = td.GaussianPulse(
            freq0=self.cfg.freq0,
            fwidth=self.cfg.fwidth,
        )
        mode_source = td.ModeSource(
            center=(-self.cfg.sim_length / 2 + 0.5, 0, 0),
            size=(0, td.inf, td.inf),
            source_time=pulse,
            direction="+",
            name="source",
        )

        mode_monitor = td.ModeMonitor(
            center=(self.cfg.sim_length / 2 - 0.5, 0, 0),
            size=(0, td.inf, td.inf),
            freqs=[self.cfg.freq0],
            name="mode",
        )

        structures = [waveguide, box]
        symmetry = (0, -1, 1) if self.cfg.use_symmetry else (0, 0, 0)
        sim = td.Simulation(
            size=(self.cfg.sim_length, 10, 2.0),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
            structures=tuple(structures),
            symmetry=symmetry,
            sources=tuple([mode_source]),
            monitors=tuple([mode_monitor]),
            shutoff=1e-4,
            run_time=100e-12,
            boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
            medium=mat_sio2,
        )
        return sim

    def add_field_monitor(self, sim):
        field_monitor = td.FieldMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[self.cfg.freq0],
            name="field",
        )
        return sim.updated_copy(monitors=sim.monitors + (field_monitor,))

    def get_full_sim(self, actions):
        sim = self.get_base_sim()
        if self.cfg.add_field_monitor:
            sim = self.add_field_monitor(sim)

        # pos_actions = actions[:60] if self.cfg.include_radius else actions
        cs, cl, rs, rl = self.get_hole_position_and_radii(actions)

        # radii = actions[60:] if self.cfg.include_radius else None
        holes = self.construct_holes(cs, cl, rs, rl)

        sim = sim.updated_copy(structures=sim.structures + (holes,))
        return sim

    def get_hole_position_and_radii(self, full_actions):
        pos_actions = full_actions[:60] if self.cfg.include_radius else full_actions
        pos_actions = pos_actions.reshape(10, 3, 2)
        cs, cl = self.full_base_lattices(taper_offset=True)
        neg_y_actions = np.stack([pos_actions[:, :, 0], -pos_actions[:, :, 1]], axis=-1)
        neg_x_actions = np.stack([-pos_actions[:, :, 0], pos_actions[:, :, 1]], axis=-1)
        neg_xy_actions = np.stack(
            [-pos_actions[:, :, 0], -pos_actions[:, :, 1]], axis=-1
        )

        cl3 = np.concatenate(
            [
                neg_y_actions[:, 0, :] + cl[:10, 3, :],
                cl[10:-10, 3, :],
                neg_xy_actions[::-1, 0, :] + cl[-10:, 3, :],
            ],
            axis=0,
        )
        cl4 = np.concatenate(
            [
                neg_y_actions[:, 1, :] + cl[:10, 4, :],
                cl[10:-10, 4, :],
                neg_xy_actions[::-1, 1, :] + cl[-10:, 4, :],
            ],
            axis=0,
        )
        cl5 = np.concatenate(
            [
                pos_actions[:, 1, :] + cl[:10, 5, :],
                cl[10:-10, 5, :],
                neg_x_actions[::-1, 1, :] + cl[-10:, 5, :],
            ],
            axis=0,
        )
        cl6 = np.concatenate(
            [
                pos_actions[:, 0, :] + cl[:10, 6, :],
                cl[10:-10, 6, :],
                neg_x_actions[::-1, 0, :] + cl[-10:, 6, :],
            ],
            axis=0,
        )
        cs4 = np.concatenate(
            [
                neg_y_actions[:, 2, :] + cs[:10, 4, :],
                cs[10:-10, 4, :],
                neg_xy_actions[::-1, 2, :] + cs[-10:, 4, :],
            ],
            axis=0,
        )
        cs5 = np.concatenate(
            [
                pos_actions[:, 2, :] + cs[:10, 5, :],
                cs[10:-10, 5, :],
                neg_x_actions[::-1, 2, :] + cs[-10:, 5, :],
            ],
            axis=0,
        )

        new_cl = np.concatenate(
            [
                cl[:, :3, :],
                cl3[:, None, :],
                cl4[:, None, :],
                cl5[:, None, :],
                cl6[:, None, :],
                cl[:, 7:, :],
            ],
            axis=1,
        )
        new_cs = np.concatenate(
            [cs[:, :4, :], cs4[:, None, :], cs5[:, None, :], cs[:, 6:, :]], axis=1
        )

        rl, rs = None, None
        if self.cfg.include_radius:
            mid_rl = np.ones_like(cl[10:-10, 0, 0]) * self.cfg.hole_radius  # type: ignore
            mid_rs = np.ones_like(cs[10:-10, 0, 0]) * self.cfg.hole_radius  # type: ignore
            rad_actions = full_actions[60:]
            rl3 = np.concatenate(
                [rad_actions[:10], mid_rl, rad_actions[:10][::-1]], axis=0
            )
            rl4 = np.concatenate(
                [rad_actions[10:20], mid_rl, rad_actions[10:20][::-1]], axis=0
            )
            rl5 = np.concatenate(
                [rad_actions[10:20], mid_rl, rad_actions[10:20][::-1]], axis=0
            )
            rl6 = np.concatenate(
                [rad_actions[:10], mid_rl, rad_actions[:10][::-1]], axis=0
            )
            rs4 = np.concatenate(
                [rad_actions[20:], mid_rs, rad_actions[20:][::-1]], axis=0
            )
            rs5 = np.concatenate(
                [rad_actions[20:], mid_rs, rad_actions[20:][::-1]], axis=0
            )

            rl = np.concatenate(
                [
                    np.ones_like(cl[:, :3, 0]) * self.cfg.hole_radius,  # type: ignore
                    rl3[:, None],
                    rl4[:, None],
                    rl5[:, None],
                    rl6[:, None],
                    np.ones_like(cl[:, :3, 0]) * self.cfg.hole_radius,  # type: ignore
                ],
                axis=1,
            )
            rs = np.concatenate(
                [
                    np.ones_like(cs[:, :4, 0]) * self.cfg.hole_radius,  # type: ignore
                    rs4[:, None],
                    rs5[:, None],
                    np.ones_like(cs[:, 6:, 0]) * self.cfg.hole_radius,  # type: ignore
                ],
                axis=1,
            )

        return new_cs, new_cl, rs, rl

    def construct_holes(
        self,
        coord_short: onp.ndarray,
        coord_long: onp.ndarray,
        rs: onp.ndarray | None,
        rl: onp.ndarray | None,
    ):
        assert (rl is None) == (rs is None)
        hole_list = []
        mat_sio2 = td.Medium(permittivity=self.cfg.n_sio2**2)
        all_coords = np.concatenate(
            (coord_short.reshape(-1, 2), coord_long.reshape(-1, 2)), axis=0
        )
        all_radii = None
        if rl is not None and rs is not None:
            all_radii = np.concatenate((rs.reshape(-1), rl.reshape(-1)), axis=0)
        for idx in range(all_coords.shape[0]):
            cur_c = all_coords[idx]
            cx, cy = cur_c
            if all_radii is None:
                cur_radius = self.cfg.hole_radius
            else:
                cur_radius = all_radii[idx]
            cur_cylinder = td.Cylinder(
                center=(cx, cy, 0),
                radius=cur_radius,
                length=self.cfg.slab_height,
                axis=2,
            )
            hole_list.append(cur_cylinder)

        holes = td.Structure(
            geometry=td.GeometryGroup(geometries=hole_list),  # type: ignore
            medium=mat_sio2,
            name="Holes",
        )
        return holes

    def full_base_lattices(self, taper_offset: bool = True):
        x_long, x_short, y_half = self.base_lattice_arrs()

        y_half_short = y_half[1::2]
        y_half_long = y_half[::2]

        y_half_short = onp.repeat(y_half_short[None, :], repeats=len(x_short), axis=0)
        y_half_long = onp.repeat(y_half_long[None, :], repeats=len(x_long), axis=0)

        x_long = onp.repeat(x_long[:, None], repeats=len(y_half), axis=1)
        x_short = onp.repeat(x_short[:, None], repeats=len(y_half), axis=1)

        if taper_offset:
            taper_offset_long = onp.linspace(
                0, self.cfg.taper_offset, self.cfg.num_cycles_taper + 1
            )[1:][::-1]
            taper_offset_short = onp.linspace(
                0, self.cfg.taper_offset, self.cfg.num_cycles_taper
            )[1:][::-1]
            num_offset_long, num_offset_short = (
                len(taper_offset_long),
                len(taper_offset_short),
            )

            y_half_short[:num_offset_short, :] += taper_offset_short[:, None]
            y_half_short[-num_offset_short:, :] += taper_offset_short[:, None][::-1]

            y_half_long[:num_offset_long, :] += taper_offset_long[:, None]
            y_half_long[-num_offset_long:, :] += taper_offset_long[:, None][::-1]

        y_short = onp.concatenate([-y_half_short[:, ::-1], y_half_short], axis=1)
        y_long = onp.concatenate([-y_half_long[:, ::-1], y_half_long], axis=1)

        coord_long = onp.stack([x_long, y_long], axis=-1)
        coord_short = onp.stack([x_short, y_short], axis=-1)

        return coord_short, coord_long

    def base_lattice_arrs(self):
        dw = self.cfg.lattice_constant
        dh = self.cfg.lattice_height
        y_half = onp.arange(0, 10) * dh + dh

        half_cycles = self.cfg.num_cycles_total // 2
        x_long = onp.arange(-half_cycles, half_cycles) * dw
        x_long = x_long - x_long.mean()

        x_short = x_long[:-1] + dw / 2
        return x_long, x_short, y_half

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
        # fixed_time_folder=f"pcr_bonni_{seed}",
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
    env_cfg = CrystalTaperEnvConfig()
    env = CrystalTaperEnv(env_cfg)

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
