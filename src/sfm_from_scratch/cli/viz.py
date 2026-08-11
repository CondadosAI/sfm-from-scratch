"""Replay saved reconstruction snapshots in Rerun and save a .rrd artifact."""


import click
import numpy as np

from sfm_from_scratch.cli.run import resolve_dataset
from sfm_from_scratch.config import OUTPUT_DIR, SNAPSHOT_DIR
from sfm_from_scratch.core.viz import log_to_rerun


def load_snapshots(label: str) -> list[dict]:
    paths = sorted((SNAPSHOT_DIR / label).glob("step_*.npz"))
    if not paths:
        raise click.ClickException(f"no snapshots under {SNAPSHOT_DIR / label} — run sfm-run first")
    return [dict(np.load(p)) for p in paths]


@click.command()
@click.option("--label", default="templeRing", show_default=True)
@click.option("--images", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--focal", type=float, default=None)
@click.option("--spawn", is_flag=True, help="Also open the interactive Rerun viewer.")
def viz(label, images, focal, spawn) -> None:
    """Log the growth timeline to Rerun (world points + camera frusta)."""
    dataset = resolve_dataset(label.removesuffix("-noba"), images, focal)
    snapshots = load_snapshots(label)
    log_to_rerun(snapshots, dataset, OUTPUT_DIR / f"{label}.rrd", spawn=spawn)


if __name__ == "__main__":
    viz()
