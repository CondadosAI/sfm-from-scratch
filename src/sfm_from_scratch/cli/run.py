"""End-to-end incremental SfM: features → match → init → grow → bundle adjust."""

import json
import time
from pathlib import Path

import click
import numpy as np
from loguru import logger
from tabulate import tabulate

from sfm_from_scratch.config import (
    BA_EVERY_N_IMAGES,
    DATA_DIR,
    OUTPUT_DIR,
    SNAPSHOT_DIR,
)
from sfm_from_scratch.core.bundle import bundle_adjust
from sfm_from_scratch.core.dataset import Dataset, load_image_folder, load_middlebury
from sfm_from_scratch.core.features import build_match_graph, extract_features
from sfm_from_scratch.core.reconstruction import Reconstruction


def resolve_dataset(label: str, images: str | None, focal: float | None) -> Dataset:
    if images is not None:
        return load_image_folder(Path(images), focal_px=focal)
    return load_middlebury(DATA_DIR, label)


@click.command()
@click.option("--dataset", "name", default="templeRing", show_default=True)
@click.option(
    "--images",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Reconstruct a bare folder of photos instead of a Middlebury dataset.",
)
@click.option("--focal", type=float, default=None, help="Focal length in px for --images.")
@click.option("--ba/--no-ba", default=True, show_default=True, help="Run bundle adjustment.")
@click.option("--ba-every", default=BA_EVERY_N_IMAGES, show_default=True)
@click.option("--max-images", default=None, type=int, help="Cap the number of input images.")
def run(name, images, focal, ba, ba_every, max_images) -> None:
    """Reconstruct a scene and save snapshots + summary artifacts to output/."""
    label = Path(images).name if images else name
    if not ba:
        label = f"{label}-noba"
    logger.add(OUTPUT_DIR / f"{label}_run.log", level="DEBUG", mode="w")

    t0 = time.perf_counter()
    dataset = resolve_dataset(name, images, focal)
    if max_images:
        dataset.views = dataset.views[:max_images]

    extract_features(dataset)
    t_feat = time.perf_counter()
    cache_name = Path(images).name if images else name
    graph = build_match_graph(dataset, cache_path=DATA_DIR / "cache" / f"{cache_name}_matches.npz")
    t_match = time.perf_counter()

    rec = Reconstruction(dataset, graph)
    rec.initialize()

    snap_dir = SNAPSHOT_DIR / label
    snap_dir.mkdir(parents=True, exist_ok=True)
    for old in snap_dir.glob("step_*.npz"):
        old.unlink()

    def save_step(step: int) -> dict:
        snap = rec.snapshot()
        np.savez_compressed(snap_dir / f"step_{step:03d}.npz", **snap)
        stats = rec.reprojection_error_stats()
        stats["n_views"] = len(rec.poses)
        return stats

    per_step = [save_step(0)]
    ba_stats: list[dict] = []
    failed: set[int] = set()
    step = 1
    while (nxt := rec.select_next_view(exclude=failed)) is not None:
        if not rec.register_view(nxt):
            failed.add(nxt)
            continue
        if ba and len(rec.poses) % ba_every == 0:
            ba_stats.append(bundle_adjust(rec))
            n_dropped = rec.filter_points()
            logger.debug(f"post-BA filter dropped {n_dropped} points")
        per_step.append(save_step(step))
        step += 1

    # Final polish: BA → filter the residual outliers → BA once more on the
    # cleaned set (a cheap version of COLMAP's iterative refine-and-filter).
    if ba:
        ba_stats.append(bundle_adjust(rec))
        rec.filter_points()
        ba_stats.append(bundle_adjust(rec))
    else:
        rec.filter_points()
    per_step.append(save_step(step))
    t_end = time.perf_counter()

    final = rec.reprojection_error_stats()
    final["n_views"] = len(rec.poses)
    summary = {
        "label": label,
        "n_input_images": len(dataset),
        "registered_images": len(rec.poses),
        "failed_images": sorted(dataset.views[i].name for i in failed),
        "bundle_adjustment": ba,
        "final": final,
        "per_step": per_step,
        "ba_stats": ba_stats,
        "timing_s": {
            "features": round(t_feat - t0, 2),
            "matching": round(t_match - t_feat, 2),
            "reconstruction": round(t_end - t_match, 2),
            "total": round(t_end - t0, 2),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"{label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    rows = [
        ["images registered", f"{len(rec.poses)}/{len(dataset)}"],
        ["3-D points", f"{final['n_points']:,}"],
        ["observations", f"{final['n_obs']:,}"],
        ["mean reprojection error", f"{final['mean_px']:.2f} px"],
        ["median reprojection error", f"{final['median_px']:.2f} px"],
        ["total runtime", f"{summary['timing_s']['total']:.1f} s"],
    ]
    print(tabulate(rows, headers=[label, "value"], tablefmt="github"))
    logger.info(f"summary saved to {summary_path}; snapshots in {snap_dir}")


if __name__ == "__main__":
    run()
