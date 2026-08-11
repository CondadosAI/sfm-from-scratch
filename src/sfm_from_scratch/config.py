"""Project-wide constants and artifact paths.

Paths resolve from the project root (found by walking up to pyproject.toml) so
every CLI works from any CWD; each is overridable via environment variable for
non-standard setups (e.g. shared data drives).
"""

import os
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA_DIR = Path(os.environ.get("SFM_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("SFM_OUTPUT_DIR", PROJECT_ROOT / "output"))
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
FRAMES_DIR = OUTPUT_DIR / "frames"

# Middlebury multi-view stereo datasets (Seitz et al., CVPR 2006).
# templeRing: 47 views on a ring, 640x480, undistorted, per-image K/R/t provided.
# Size verified 2026-08-10: 11,707,443 bytes.
MIDDLEBURY_BASE_URL = "https://vision.middlebury.edu/mview/data/data"
DATASETS = {
    "templeRing": f"{MIDDLEBURY_BASE_URL}/templeRing.zip",
    "dinoRing": f"{MIDDLEBURY_BASE_URL}/dinoRing.zip",
}

# --- Pipeline defaults -------------------------------------------------------
# SIFT: default nfeatures=0 keeps every detection; the plaster temple is
# low-texture, so we do not want to throw responses away.
SIFT_NFEATURES = 0

# Lowe ratio test: 0.75 per the SIFT paper's recommended range.
LOWE_RATIO = 0.75

# Minimum ratio-test matches for a pair to enter the match graph at all.
MIN_PAIR_MATCHES = 30

# RANSAC threshold (px) for essential-matrix estimation and PnP.
RANSAC_THRESHOLD_PX = 1.5
PNP_RANSAC_THRESHOLD_PX = 4.0

# Point filters: observations must reproject within this error (px), lie in
# front of the cameras, and subtend a minimum triangulation angle (degrees) —
# small-parallax points have unbounded depth uncertainty.
MAX_REPROJ_ERROR_PX = 4.0
MIN_TRIANGULATION_ANGLE_DEG = 1.5

# Initialization: among well-matched pairs, require a median triangulation
# angle above this so the first two views carry real parallax.
INIT_MIN_MEDIAN_ANGLE_DEG = 4.0

# Bundle adjustment: run a local/global BA every N registered images, plus a
# final global pass.
BA_EVERY_N_IMAGES = 5
