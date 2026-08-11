"""Sparse bundle adjustment with scipy.optimize.least_squares.

Follows the SciPy Cookbook "large-scale bundle adjustment" pattern
(https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html), adapted to
our setup: intrinsics K are known and stay FIXED; we refine camera poses
(rvec, tvec — 6 params each) and 3-D points (3 params each), minimizing pixel
reprojection error over every observation simultaneously.

The first registered camera is excluded from the parameters — it anchors the
world frame (6 of the 7 gauge freedoms; the 7th, global scale, is left free and
is harmless to the least-squares step).
"""

import time

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from sfm_from_scratch.core.reconstruction import Reconstruction


def _gather(rec: Reconstruction):
    """Flatten the reconstruction into the arrays BA operates on."""
    cam_ids = list(rec.registration_order)
    cam_local = {c: i for i, c in enumerate(cam_ids)}
    pt_ids = [pid for pid, pt in rec.points.items() if len(pt.obs) >= 2]
    pt_local = {p: i for i, p in enumerate(pt_ids)}

    obs_cam, obs_pt, obs_xy = [], [], []
    for pid in pt_ids:
        for v, kp in rec.points[pid].obs.items():
            obs_cam.append(cam_local[v])
            obs_pt.append(pt_local[pid])
            obs_xy.append(rec.dataset.views[v].keypoints[kp].pt)
    K_per_cam = np.stack([rec.dataset.views[c].K for c in cam_ids])
    return (
        cam_ids,
        pt_ids,
        np.array(obs_cam),
        np.array(obs_pt),
        np.array(obs_xy, dtype=np.float64),
        K_per_cam,
    )


def _project(cam_params: np.ndarray, X: np.ndarray, obs_cam, obs_pt, K_per_cam) -> np.ndarray:
    """Project every observed point through its camera → (n_obs, 2) pixels."""
    n_cams = len(K_per_cam)
    R = np.empty((n_cams, 3, 3))
    for c in range(n_cams):
        R[c], _ = cv2.Rodrigues(cam_params[c, :3])
    Xc = np.einsum("oij,oj->oi", R[obs_cam], X[obs_pt]) + cam_params[obs_cam, 3:6]
    uvw = np.einsum("oij,oj->oi", K_per_cam[obs_cam], Xc)
    return uvw[:, :2] / uvw[:, 2:3]


def bundle_adjust(rec: Reconstruction, max_nfev: int = 50) -> dict[str, float]:
    """Refine all registered poses + all points in place; returns before/after stats."""
    cam_ids, pt_ids, obs_cam, obs_pt, obs_xy, K_per_cam = _gather(rec)
    n_cams, n_pts, n_obs = len(cam_ids), len(pt_ids), len(obs_xy)

    cam0 = np.empty((n_cams, 6))
    for i, c in enumerate(cam_ids):
        R, t = rec.poses[c]
        cam0[i, :3] = cv2.Rodrigues(R)[0].ravel()
        cam0[i, 3:6] = t
    X0 = np.array([rec.points[p].xyz for p in pt_ids])

    # Free parameters: cameras 1..n-1 (camera 0 anchors the frame) + all points.
    def unpack(params):
        cams = cam0.copy()
        cams[1:] = params[: (n_cams - 1) * 6].reshape(n_cams - 1, 6)
        pts = params[(n_cams - 1) * 6 :].reshape(n_pts, 3)
        return cams, pts

    def residuals(params):
        cams, pts = unpack(params)
        return (_project(cams, pts, obs_cam, obs_pt, K_per_cam) - obs_xy).ravel()

    x0 = np.hstack([cam0[1:].ravel(), X0.ravel()])

    # Jacobian sparsity: each residual pair touches one camera's 6 params and
    # one point's 3 params — the structure that makes 100k-observation BA cheap.
    sparsity = lil_matrix((2 * n_obs, x0.size), dtype=int)
    rows = np.arange(n_obs)
    for k in range(6):
        free = obs_cam > 0
        cols = (obs_cam[free] - 1) * 6 + k
        sparsity[2 * rows[free], cols] = 1
        sparsity[2 * rows[free] + 1, cols] = 1
    for k in range(3):
        cols = (n_cams - 1) * 6 + obs_pt * 3 + k
        sparsity[2 * rows, cols] = 1
        sparsity[2 * rows + 1, cols] = 1

    before = np.linalg.norm(residuals(x0).reshape(-1, 2), axis=1)
    t0 = time.perf_counter()
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        x_scale="jac",
        ftol=1e-4,
        max_nfev=max_nfev,
    )
    elapsed = time.perf_counter() - t0
    after = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)

    cams, pts = unpack(result.x)
    for i, c in enumerate(cam_ids):
        R, _ = cv2.Rodrigues(cams[i, :3])
        rec.poses[c] = (R, cams[i, 3:6].copy())
    for i, p in enumerate(pt_ids):
        rec.points[p].xyz = pts[i].copy()

    stats = {
        "n_cams": n_cams,
        "n_points": n_pts,
        "n_obs": n_obs,
        "mean_before_px": float(before.mean()),
        "mean_after_px": float(after.mean()),
        "median_before_px": float(np.median(before)),
        "median_after_px": float(np.median(after)),
        "seconds": elapsed,
    }
    logger.info(
        f"BA: {n_cams} cams, {n_pts} pts, {n_obs} obs — mean reproj "
        f"{stats['mean_before_px']:.2f} → {stats['mean_after_px']:.2f} px in {elapsed:.1f}s"
    )
    return stats
