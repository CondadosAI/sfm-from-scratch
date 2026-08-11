"""Reference check: run pycolmap on the same images and tabulate both results.

pycolmap is COLMAP's real incremental mapper — the industrial version of what
our few hundred lines implement. It is allowed to self-calibrate (we don't feed
it the known K): it serves as the credibility reference, not a controlled
benchmark.
"""

import csv
import json
import shutil
import time

import click
from loguru import logger
from tabulate import tabulate

from sfm_from_scratch.config import DATA_DIR, OUTPUT_DIR


@click.command()
@click.option("--dataset", "name", default="templeRing", show_default=True)
def check(name: str) -> None:
    """Run pycolmap end-to-end and write the comparison table to output/."""
    import pycolmap

    src_dir = DATA_DIR / name
    work = OUTPUT_DIR / "colmap" / name
    image_dir = work / "images"
    if image_dir.exists():
        shutil.rmtree(work)
    image_dir.mkdir(parents=True)
    for p in sorted(src_dir.glob("*.png")):
        shutil.copy(p, image_dir / p.name)  # pngs only; keep the _par.txt out

    db = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir()
    t0 = time.perf_counter()
    pycolmap.extract_features(db, image_dir)
    pycolmap.match_exhaustive(db)
    maps = pycolmap.incremental_mapping(db, image_dir, sparse)
    elapsed = time.perf_counter() - t0
    if not maps:
        raise click.ClickException("pycolmap produced no reconstruction")
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    logger.info(f"pycolmap: {best.summary()}")

    ours_path = OUTPUT_DIR / f"{name}_summary.json"
    if not ours_path.exists():
        raise click.ClickException(f"{ours_path} missing — run sfm-run first")
    ours = json.loads(ours_path.read_text())

    n_images = ours["n_input_images"]
    rows = [
        [
            "mini-SfM (this repo)",
            f"{ours['registered_images']}/{n_images}",
            f"{ours['final']['n_points']:,}",
            f"{ours['final']['n_obs'] / max(ours['final']['n_points'], 1):.1f}",
            f"{ours['final']['mean_px']:.2f}",
            f"{ours['timing_s']['total']:.0f}",
        ],
        [
            f"pycolmap {pycolmap.__version__}",
            f"{best.num_reg_images()}/{n_images}",
            f"{best.num_points3D():,}",
            f"{best.compute_mean_track_length():.1f}",
            f"{best.compute_mean_reprojection_error():.2f}",
            f"{elapsed:.0f}",
        ],
    ]
    headers = [name, "registered", "3-D points", "track len", "mean reproj (px)", "time (s)"]
    table = tabulate(rows, headers=headers, tablefmt="github")
    print(table)

    (OUTPUT_DIR / "results.md").write_text(
        f"# {name}: mini-SfM vs pycolmap\n\n{table}\n\n"
        "Note: pycolmap self-calibrates (SIMPLE_RADIAL) while mini-SfM uses the "
        "dataset's known K; reprojection errors are therefore not computed against "
        "identical camera models.\n"
    )
    with (OUTPUT_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info(f"wrote {OUTPUT_DIR / 'results.md'} and results.csv")


if __name__ == "__main__":
    check()
