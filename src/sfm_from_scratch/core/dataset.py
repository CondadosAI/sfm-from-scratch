"""Dataset loading: images + calibration.

Middlebury ring datasets ship a ``*_par.txt`` with per-image intrinsics K and
ground-truth extrinsics R, t (Seitz et al., CVPR 2006 format):

    <n_images>
    <name> k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 ... r33 t1 t2 t3

We use K as given (the images are already undistorted) and keep the
ground-truth poses ONLY for evaluation — the pipeline never reads them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


@dataclass
class View:
    """One input image plus its intrinsics (and optional GT pose)."""

    name: str
    path: Path
    K: np.ndarray  # (3,3) float64
    R_gt: np.ndarray | None = None  # (3,3) world->camera
    t_gt: np.ndarray | None = None  # (3,)
    keypoints: list = field(default_factory=list)  # cv2.KeyPoint
    descriptors: np.ndarray | None = None

    def load_bgr(self) -> np.ndarray:
        img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.path)
        return img

    def load_gray(self) -> np.ndarray:
        return cv2.cvtColor(self.load_bgr(), cv2.COLOR_BGR2GRAY)

    @property
    def kp_xy(self) -> np.ndarray:
        """Keypoint pixel coordinates as an (N,2) float64 array."""
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float64)


@dataclass
class Dataset:
    name: str
    views: list[View]

    def __len__(self) -> int:
        return len(self.views)


def load_middlebury(root: Path, name: str) -> Dataset:
    """Load a Middlebury ring dataset extracted under ``root/name``."""
    ddir = root / name
    par_files = sorted(ddir.glob("*_par.txt"))
    if not par_files:
        raise FileNotFoundError(f"no *_par.txt under {ddir} — run sfm-download first")
    lines = par_files[0].read_text().strip().splitlines()
    n = int(lines[0])
    views: list[View] = []
    for line in lines[1 : n + 1]:
        parts = line.split()
        img_name, vals = parts[0], np.array([float(v) for v in parts[1:]])
        K = vals[0:9].reshape(3, 3)
        R = vals[9:18].reshape(3, 3)
        t = vals[18:21]
        views.append(View(name=img_name, path=ddir / img_name, K=K, R_gt=R, t_gt=t))
    logger.info(f"{name}: {len(views)} views loaded from {par_files[0].name}")
    return Dataset(name=name, views=views)


def load_image_folder(folder: Path, focal_px: float | None = None) -> Dataset:
    """Load a bare folder of photos (the phone-capture path).

    Without calibration we seed a shared pinhole K: principal point at the
    image center and focal length either given or approximated as
    1.2 * max(width, height) — a common wide-ish smartphone prior. This is a
    deliberately honest hack; the article discusses its fragility.
    """
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise FileNotFoundError(f"no images in {folder}")
    sample = cv2.imread(str(paths[0]))
    h, w = sample.shape[:2]
    f = focal_px if focal_px is not None else 1.2 * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    logger.info(f"{folder.name}: {len(paths)} images, seeded K with f={f:.0f}px ({w}x{h})")
    return Dataset(name=folder.name, views=[View(name=p.name, path=p, K=K.copy()) for p in paths])
