"""Two-view geometry: essential matrix, pose recovery, triangulation, filters.

Conventions: poses are world→camera, x_cam = R @ X_world + t. A camera's
projection matrix is P = K [R | t]; its centre in world coords is C = -Rᵀ t.
"""

import cv2
import numpy as np

from sfm_from_scratch.config import RANSAC_THRESHOLD_PX


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t.reshape(3)


def estimate_relative_pose(
    pts_a: np.ndarray, pts_b: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Essential matrix + the ONE physical (R, t) of camera b w.r.t. camera a.

    E decomposes into four (R, t) candidates; recoverPose picks the one that
    puts the most triangulated points in front of BOTH cameras (the cheirality
    check). ‖t‖ is set to 1 — monocular SfM has no absolute scale.

    Returns (R, t, inlier_mask over the input correspondences).
    """
    E, mask_e = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=RANSAC_THRESHOLD_PX
    )
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask_e.copy())
    return R, t.reshape(3), mask_pose.ravel().astype(bool)


def triangulate(
    P_a: np.ndarray, P_b: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray:
    """DLT triangulation of matched pixel coords → (N,3) world points.

    cv2.triangulatePoints solves the linear (algebraic) least-squares system,
    not the reprojection-optimal one — good enough as a seed; bundle adjustment
    polishes it (H&Z ch. 12 discusses the optimal alternatives).
    """
    Xh = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)  # (4,N) homogeneous
    return (Xh[:3] / Xh[3]).T


def depths(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """z-coordinate of world points in a camera's frame (positive = in front)."""
    return (X @ R.T + t)[:, 2]


def reprojection_errors(
    X: np.ndarray, xy: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Per-point pixel distance between projected world points and observations."""
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, t.reshape(3), K, None)
    return np.linalg.norm(proj.reshape(-1, 2) - xy, axis=1)


def triangulation_angles(
    X: np.ndarray, center_a: np.ndarray, center_b: np.ndarray
) -> np.ndarray:
    """Angle (degrees) subtended at each point by the two camera centres.

    A point seen with ~0° parallax lies on (nearly) parallel rays: its depth is
    unconstrained and triangulation is numerically explosive. We threshold on
    this angle everywhere.
    """
    ray_a = center_a - X
    ray_b = center_b - X
    cosang = np.sum(ray_a * ray_b, axis=1) / (
        np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-12
    )
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
