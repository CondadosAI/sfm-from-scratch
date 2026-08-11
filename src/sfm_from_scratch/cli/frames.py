"""Render growth-animation frames (matplotlib, headless) and encode mp4/GIF."""

import click

from sfm_from_scratch.cli.run import resolve_dataset
from sfm_from_scratch.cli.viz import load_snapshots
from sfm_from_scratch.config import FRAMES_DIR, OUTPUT_DIR
from sfm_from_scratch.core.viz import encode_video, render_growth_frames


@click.command()
@click.option("--label", default="templeRing", show_default=True)
@click.option("--images", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--focal", type=float, default=None)
@click.option("--fps", default=6, show_default=True)
@click.option("--orbit", default=60.0, show_default=True, help="Total orbit sweep (degrees).")
@click.option("--gif/--no-gif", default=True, show_default=True, help="Also write a .gif.")
def frames(label, images, focal, fps, orbit, gif) -> None:
    """PNG frames per registration step → output/<label>_{scene,closeup}.mp4 (+ .gif)."""
    dataset = resolve_dataset(label.removesuffix("-noba"), images, focal)
    snapshots = load_snapshots(label)
    for view in ("scene", "closeup"):
        paths = render_growth_frames(
            snapshots, dataset, FRAMES_DIR / f"{label}_{view}", orbit_degrees=orbit, view=view
        )
        encode_video(paths, OUTPUT_DIR / f"{label}_{view}.mp4", fps=fps)
        if gif:
            encode_video(paths, OUTPUT_DIR / f"{label}_{view}.gif", fps=fps)


if __name__ == "__main__":
    frames()
