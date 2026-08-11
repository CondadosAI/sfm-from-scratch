# sfm-from-scratch

Incremental Structure from Motion in a few hundred readable lines: SIFT
features → ratio-test + RANSAC matching → essential-matrix initialization →
PnP growth → sparse bundle adjustment (SciPy). pycolmap runs as the industrial
reference; Rerun and matplotlib render the reconstruction growing image by
image.

Companion code for the CondadosAI article on Structure from Motion.

## Layout

```
sfm-from-scratch/
├── pyproject.toml
├── src/sfm_from_scratch/
│   ├── config.py              # paths + every pipeline threshold, documented
│   ├── core/
│   │   ├── dataset.py         # Middlebury ring loader (+ bare photo folders)
│   │   ├── features.py        # SIFT, ratio test, verified match graph
│   │   ├── geometry.py        # E, recoverPose, triangulation, filters
│   │   ├── reconstruction.py  # the incremental loop (init → PnP → grow)
│   │   ├── bundle.py          # sparse BA (scipy least_squares)
│   │   └── viz.py             # Rerun logging + matplotlib growth frames
│   └── cli/                   # one click command per file (entry points below)
├── data/                      # datasets + match cache (gitignored)
└── output/                    # snapshots, summaries, tables, videos (artifacts)
```

## Setup

```bash
uv sync
```

## Pipeline

```bash
uv run sfm-download                  # Middlebury templeRing (11 MB, 47 views)
uv run sfm-run                       # full reconstruction + snapshots + summary
uv run sfm-run --no-ba               # ablation: same pipeline without bundle adjustment
uv run sfm-viz --spawn               # interactive Rerun replay (saves output/templeRing.rrd)
uv run sfm-frames                    # growth animation → output/templeRing.mp4/.gif
uv run sfm-check                     # pycolmap reference → output/results.md/.csv
```

Your own photos (~30 images orbiting a textured object, fixed zoom/exposure):

```bash
uv run sfm-run --images path/to/photos --focal 3200   # focal in px, optional
uv run sfm-frames --label photos --images path/to/photos
```

## Data

Middlebury multi-view datasets (templeRing/dinoRing) by Seitz, Curless, Diebel,
Scharstein & Szeliski, *A Comparison and Evaluation of Multi-View Stereo
Reconstruction Algorithms*, CVPR 2006 — <https://vision.middlebury.edu/mview/>.
The images ship with per-view calibration; the pipeline uses K only (ground-truth
poses are never read during reconstruction).

## License

The code in this repository is licensed under the
[Apache License 2.0](LICENSE).

**That covers this code and nothing else.** The Middlebury multi-view datasets
(`templeRing` and friends) are downloaded at run time from
vision.middlebury.edu, which states no formal licence and asks that you cite
Seitz et al., CVPR 2006. They are not redistributed here. pycolmap and COLMAP
carry their own licence upstream. Check the terms yourself before any commercial
use.
