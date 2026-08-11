"""Visualization: Rerun logging (interactive) + matplotlib frame rendering (GIF/mp4).

Two backends on purpose: Rerun gives the explorable artifact (.rrd) and the
best interactive experience; matplotlib gives deterministic, headless-safe
frames for the article's growth animation without depending on the viewer's
experimental screenshot API.
"""

from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfm_from_scratch.core.dataset import Dataset  # noqa: E402


def camera_centers(snapshot: dict) -> np.ndarray:
    """World-space camera centres C = -Rᵀ t for every registered view."""
    return np.einsum("nij,nj->ni", -snapshot["R"].transpose(0, 2, 1), snapshot["t"])


def log_to_rerun(
    snapshots: list[dict], dataset: Dataset, rrd_path: Path, spawn: bool = False
) -> None:
    """Replay the reconstruction growth on a Rerun timeline and save a .rrd."""
    import rerun as rr

    rr.init("sfm_from_scratch", spawn=spawn)
    for step, snap in enumerate(snapshots):
        rr.set_time("registration", sequence=step)
        rr.log(
            "world/points",
            rr.Points3D(snap["xyz"], colors=snap["rgb"], radii=0.003),
        )
        for k, view_idx in enumerate(snap["view_indices"]):
            view = dataset.views[int(view_idx)]
            R, t = snap["R"][k], snap["t"][k]
            entity = f"world/cameras/{view.name}"
            rr.log(
                entity,
                rr.Transform3D(translation=-R.T @ t, mat3x3=R.T),
            )
            h, w = 480, 640  # logged once via Pinhole resolution below
            rr.log(
                entity,
                rr.Pinhole(image_from_camera=view.K, resolution=[w, h], image_plane_distance=0.1),
            )
    rr.save(str(rrd_path))
    logger.info(f"Rerun recording saved to {rrd_path}")


def _draw_frustum(ax, K: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float, color: str):
    """Wireframe pyramid for one camera pose (world→camera R, t)."""
    C = -R.T @ t
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rays = np.linalg.inv(K) @ np.hstack([corners_px, np.ones((4, 1))]).T  # camera frame
    corners_world = (R.T @ (rays * scale)).T + C
    for corner in corners_world:
        ax.plot(*zip(C, corner, strict=True), color=color, linewidth=0.6)
    loop = np.vstack([corners_world, corners_world[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=0.6)


def _canonical_alignment(final: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rotation A + centroid c mapping the reconstruction to a canonical frame.

    The world frame is whatever the initial camera pair happened to be, so the
    scene renders at an arbitrary tilt. We fit a plane to the camera centres
    (SVD): its normal is the ring axis — the scene's natural "up". Sign is
    chosen to agree with the average camera up-vector (−Rᵀ·ŷ, since image y
    points down). A maps X → A @ (X − c) with up along +z (matplotlib's up).
    """
    centers = camera_centers(final)
    c = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - c)
    normal = vt[2]
    up_hint = -final["R"].transpose(0, 2, 1)[:, :, 1].mean(axis=0)
    if np.dot(normal, up_hint) < 0:
        normal = -normal
    A = np.vstack([vt[0], np.cross(normal, vt[0]), normal])
    return A, c


def render_growth_frames(
    snapshots: list[dict],
    dataset: Dataset,
    out_dir: Path,
    orbit_degrees: float = 60.0,
    elev: float = 18.0,
    hold_last: int = 8,
    view: str = "scene",
) -> list[Path]:
    """One PNG per registration step: colored cloud + frusta, slow orbit.

    Everything is drawn in the canonical (ring-up) frame; axis limits come from
    the FINAL snapshot so the framing doesn't jump as the cloud grows; the last
    frame is repeated so the loop rests on the result. view="scene" frames the
    whole camera ring (the hero shot); view="closeup" frames the object cloud.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final = snapshots[-1]
    A, c = _canonical_alignment(final)

    pts_f = (final["xyz"] - c) @ A.T
    cams_f = (camera_centers(final) - c) @ A.T
    if view == "scene":
        lo = np.minimum(np.percentile(pts_f, 2, axis=0), cams_f.min(axis=0))
        hi = np.maximum(np.percentile(pts_f, 98, axis=0), cams_f.max(axis=0))
        margin, cam_frac = 1.05, 0.10
    else:
        lo = np.percentile(pts_f, 1, axis=0)
        hi = np.percentile(pts_f, 99, axis=0)
        margin, cam_frac = 1.20, 0.16
    center = (lo + hi) / 2
    half = float(np.max(hi - lo)) / 2 * margin
    cam_scale = half * cam_frac

    paths: list[Path] = []
    n = len(snapshots)
    for step, snap in enumerate(snapshots):
        fig = plt.figure(figsize=(8, 6), dpi=120)
        ax = fig.add_subplot(projection="3d")
        pts = (snap["xyz"] - c) @ A.T
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], c=snap["rgb"] / 255.0, s=1.0, depthshade=False
        )
        for k, view_idx in enumerate(snap["view_indices"]):
            # Same world transform for the cameras: R' = R Aᵀ, t' = R c + t.
            R, t = snap["R"][k], snap["t"][k]
            is_new = k >= len(snap["view_indices"]) - 1
            _draw_frustum(
                ax,
                dataset.views[int(view_idx)].K,
                R @ A.T,
                R @ c + t,
                cam_scale,
                color="#e5484d" if is_new else "#3b82f6",
            )
        for axis, cc in zip("xyz", center, strict=True):
            getattr(ax, f"set_{axis}lim")(cc - half, cc + half)
        ax.view_init(elev=elev, azim=-60 + orbit_degrees * step / max(n - 1, 1))
        ax.set_axis_off()
        ax.set_title(
            f"views {len(snap['view_indices']):2d}/{len(dataset)}   "
            f"points {len(snap['xyz']):,}",
            fontsize=10,
            family="monospace",
        )
        fig.tight_layout(pad=0.1)
        path = out_dir / f"step_{step:03d}.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        paths.append(path)
    paths.extend([paths[-1]] * hold_last)
    logger.info(f"{len(paths)} frames rendered to {out_dir}")
    return paths


def encode_video(frame_paths: list[Path], out_path: Path, fps: int = 6) -> None:
    """Assemble frames into an .mp4 (H.264) or .gif, by extension."""
    import imageio.v3 as iio

    frames = [iio.imread(p) for p in frame_paths]
    if out_path.suffix == ".gif":
        iio.imwrite(out_path, frames, duration=int(1000 / fps), loop=0)
    else:
        iio.imwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
    logger.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
