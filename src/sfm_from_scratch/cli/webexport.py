"""Export a reconstruction as a compact binary the blog's 3-D viewer can fetch.

The viewer needs three things: where the points are, what colour they are, and
when each one was created (so the reader can watch the reconstruction grow).
Shipping the 47 raw snapshots would cost 2.7 MiB. Because the reconstruction is
append-only — point *i* keeps index *i* for the rest of the run, verified across
all 47 snapshots — the same animation fits in one final cloud plus the point count
at each step, about 105 KiB. "The points that existed at step k" is then just the
first ``points_per_step[k]`` of the buffer, which a renderer draws with a draw
range and no per-point bookkeeping.

The tradeoff that buys: every point is drawn at its **final, bundle-adjusted**
position, so the animation shows coverage growing, not bundle adjustment
settling. Points move by up to 1.27 scene units during BA (the temple spans
about 11 units), so this is a real simplification and the viewer caption says so.

Layout of ``sparse.bin`` — one buffer, read with typed-array views at the byte
offsets in ``sparse.json``:

    xyz     float32   n_points * 3
    rgb     uint8     n_points * 3
    cam_R   float32   n_cameras * 9   (row-major; P = K[R|t])
    cam_t   float32   n_cameras * 3
"""

import json
from pathlib import Path

import click
import numpy as np
from loguru import logger

from sfm_from_scratch.config import OUTPUT_DIR, SNAPSHOT_DIR


def _robust_frame(xyz: np.ndarray) -> tuple[list[float], float]:
    """Return a (center, radius) that frames the *bulk* of the cloud.

    Triangulation always leaves a few points far behind the cameras. Framing on
    min/max would zoom out until the temple is three pixels wide, so the centre
    is the median and the radius covers the 1st–99th percentile.
    """
    low = np.percentile(xyz, 1, axis=0)
    high = np.percentile(xyz, 99, axis=0)
    center = np.median(xyz, axis=0)
    radius = float(np.linalg.norm(high - low) / 2.0)
    return [float(v) for v in center], radius


@click.command()
@click.option("--dataset", default="templeRing", show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write sparse.bin + sparse.json (default: output/web/<dataset>).",
)
def webexport(dataset: str, output_dir: Path | None) -> None:
    """Pack the reconstruction snapshots into sparse.bin + sparse.json."""
    snapshots = sorted((SNAPSHOT_DIR / dataset).glob("step_*.npz"))
    if not snapshots:
        raise click.ClickException(
            f"No snapshots in {SNAPSHOT_DIR / dataset}. Run: uv run sfm-run --dataset {dataset}"
        )

    counts = []
    views_per_step = []
    for path in snapshots:
        with np.load(path) as data:
            counts.append(len(data["xyz"]))
            views_per_step.append(len(data["view_indices"]))

    # Guard the assumption this whole format rests on. If a future change to the
    # reconstruction reorders points, the growth animation would silently show
    # the wrong points appearing, so fail loudly instead.
    if counts != sorted(counts):
        raise click.ClickException(
            "Point counts are not monotonic across snapshots; the append-only "
            "assumption behind points_per_step no longer holds."
        )

    with np.load(snapshots[-1]) as final:
        xyz = final["xyz"].astype(np.float32)
        rgb = final["rgb"].astype(np.uint8)
        cam_r = final["R"].astype(np.float32)
        cam_t = final["t"].astype(np.float32)
        view_indices = final["view_indices"].astype(int).tolist()

    n_points = len(xyz)
    if counts[-1] != n_points:
        raise click.ClickException("Final snapshot disagrees with its own point count.")

    destination = output_dir or (OUTPUT_DIR / "web" / dataset)
    destination.mkdir(parents=True, exist_ok=True)

    blob = b"".join([xyz.tobytes(), rgb.tobytes(), cam_r.tobytes(), cam_t.tobytes()])
    (destination / "sparse.bin").write_bytes(blob)

    center, radius = _robust_frame(xyz)
    offset = 0
    layout = {}
    for name, array in [
        ("xyz", xyz),
        ("rgb", rgb),
        ("cam_R", cam_r),
        ("cam_t", cam_t),
    ]:
        layout[name] = {"offset": offset, "length": int(array.size)}
        offset += array.nbytes

    manifest = {
        "dataset": dataset,
        "n_points": int(n_points),
        "n_cameras": int(len(cam_r)),
        "n_steps": len(counts),
        "points_per_step": [int(c) for c in counts],
        # How many cameras were registered at each step, so the viewer can reveal
        # frusta in the order the pipeline actually added them.
        "views_per_step": [int(v) for v in views_per_step],
        "view_indices": view_indices,
        "center": center,
        "radius": radius,
        "layout": layout,
        "note": (
            "Points are stored in creation order at their final bundle-adjusted "
            "positions, so the first points_per_step[k] entries are the cloud as it "
            "stood at step k — but at their final, not their then-current, positions."
        ),
    }
    (destination / "sparse.json").write_text(json.dumps(manifest, indent=2))

    logger.success(
        f"{destination}: {n_points:,} points, {len(cam_r)} cameras, "
        f"{len(counts)} steps, {len(blob) / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    webexport()
