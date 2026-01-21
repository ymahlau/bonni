from pathlib import Path
import jax
from matplotlib import pyplot as plt
import seaborn


def plot_info(
    stacked_info: dict[str, jax.Array],
    iter: int,
    directory: Path,
):
    for k, v in stacked_info.items():
        plt.clf()
        seaborn.set_theme(style='whitegrid')
        plt.plot(v)
        plt.tight_layout()
        plt.savefig(directory / f"{k}_{iter}.png")
