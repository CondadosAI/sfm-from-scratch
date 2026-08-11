"""Incremental reconstruction: two-view seed, then grow by PnP + triangulation.

This is the same skeleton COLMAP implements industrially (Schönberger & Frahm,
CVPR 2016), reduced to its readable core:

    init two views → repeat { register best next view (PnP) → triangulate new
    points → filter → occasionally bundle-adjust } → final BA.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger

from sfm_from_scratch.config import (
    INIT_MIN_MEDIAN_ANGLE_DEG,
    MAX_REPROJ_ERROR_PX,
    MIN_TRIANGULATION_ANGLE_DEG,
    PNP_RANSAC_THRESHOLD_PX,
)
from sfm_from_scratch.core.dataset import Dataset
from sfm_from_scratch.core.geometry import (
    camera_center,
    depths,
    estimate_relative_pose,
    projection_matrix,
    reprojection_errors,
    triangulate,
    triangulation_angles,
)


@dataclass
class Point3D:
    xyz: np.ndarray  # (3,)
    color: np.ndarray  # (3,) uint8 RGB, sampled at first observation
    obs: dict[int, int] = field(default_factory=dict)  # view_idx -> keypoint_idx


class Reconstruction:
    """Holds poses + points and implements the incremental loop."""

    def __init__(self, dataset: Dataset, graph: dict[tuple[int, int], np.ndarray]):
        self.dataset = dataset
        self.graph = graph
        self.poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # idx -> (R, t)
        self.points: dict[int, Point3D] = {}
        self.point_for_kp: dict[tuple[int, int], int] = {}  # (view, kp) -> point id
        self.registration_order: list[int] = []
        self._next_point_id = 0
        self._colors: dict[int, np.ndarray] = {}  # cached RGB images for color sampling

    # ---------------------------------------------------------------- helpers

    def matches_between(self, a: int, b: int) -> np.ndarray:
        """Verified matches oriented as (kp_a, kp_b), regardless of storage order."""
        if (a, b) in self.graph:
            return self.graph[(a, b)]
        if (b, a) in self.graph:
            return self.graph[(b, a)][:, ::-1]
        return np.empty((0, 2), dtype=np.int64)

    def _rgb(self, view_idx: int) -> np.ndarray:
        if view_idx not in self._colors:
            self._colors[view_idx] = cv2.cvtColor(
                self.dataset.views[view_idx].load_bgr(), cv2.COLOR_BGR2RGB
            )
        return self._colors[view_idx]

    def _add_point(self, xyz: np.ndarray, obs: dict[int, int]) -> int:
        view_idx, kp_idx = next(iter(obs.items()))
        x, y = self.dataset.views[view_idx].keypoints[kp_idx].pt
        color = self._rgb(view_idx)[int(round(y)), int(round(x))].copy()
        pid = self._next_point_id
        self._next_point_id += 1
        self.points[pid] = Point3D(xyz=xyz.copy(), color=color, obs=dict(obs))
        for key in obs.items():
            self.point_for_kp[key] = pid
        return pid

    def _remove_point(self, pid: int) -> None:
        for key in self.points[pid].obs.items():
            self.point_for_kp.pop(key, None)
        del self.points[pid]

    # ------------------------------------------------------------ two-view init

    def select_init_pair(self) -> tuple[int, int]:
        """Best seed pair: many verified matches AND real parallax.

        A pair related by (near-)pure rotation matches beautifully but
        triangulates nothing — the classic way to doom a reconstruction. We
        score the top-matched pairs by median triangulation angle and take the
        first that clears the threshold.
        """
        ranked = sorted(self.graph, key=lambda p: len(self.graph[p]), reverse=True)
        best, best_angle = ranked[0], -1.0
        for pair in ranked[:20]:
            a, b = pair
            m = self.graph[pair]
            xy_a = self.dataset.views[a].kp_xy[m[:, 0]]
            xy_b = self.dataset.views[b].kp_xy[m[:, 1]]
            K = self.dataset.views[a].K
            R, t, inl = estimate_relative_pose(xy_a, xy_b, K)
            if inl.sum() < 50:
                continue
            P_a = projection_matrix(K, np.eye(3), np.zeros(3))
            P_b = projection_matrix(self.dataset.views[b].K, R, t)
            X = triangulate(P_a, P_b, xy_a[inl], xy_b[inl])
            ang = triangulation_angles(X, np.zeros(3), camera_center(R, t))
            med = float(np.median(ang))
            if med >= INIT_MIN_MEDIAN_ANGLE_DEG:
                logger.info(
                    f"init pair {a},{b}: {inl.sum()} inliers, median angle {med:.1f} deg"
                )
                return pair
            if med > best_angle:
                best, best_angle = pair, med
        logger.warning(
            f"no pair clears {INIT_MIN_MEDIAN_ANGLE_DEG} deg; using best ({best_angle:.1f})"
        )
        return best

    def initialize(self, pair: tuple[int, int] | None = None) -> None:
        a, b = pair if pair is not None else self.select_init_pair()
        va, vb = self.dataset.views[a], self.dataset.views[b]
        m = self.matches_between(a, b)
        xy_a, xy_b = va.kp_xy[m[:, 0]], vb.kp_xy[m[:, 1]]
        R, t, inl = estimate_relative_pose(xy_a, xy_b, va.K)

        # Gauge freedom: camera a IS the world origin, and ‖t‖=1 IS the unit of
        # length. Any other choice differs by a similarity transform.
        self.poses[a] = (np.eye(3), np.zeros(3))
        self.poses[b] = (R, t)
        self.registration_order += [a, b]

        m, xy_a, xy_b = m[inl], xy_a[inl], xy_b[inl]
        X = triangulate(
            projection_matrix(va.K, *self.poses[a]),
            projection_matrix(vb.K, *self.poses[b]),
            xy_a,
            xy_b,
        )
        keep = self._triangulation_filter(X, a, b, xy_a, xy_b)
        for k in np.flatnonzero(keep):
            self._add_point(X[k], {a: int(m[k, 0]), b: int(m[k, 1])})
        logger.info(f"initialized with views {a},{b}: {keep.sum()}/{len(X)} points kept")

    def _triangulation_filter(
        self, X: np.ndarray, a: int, b: int, xy_a: np.ndarray, xy_b: np.ndarray
    ) -> np.ndarray:
        """Cheirality + reprojection + parallax checks for freshly triangulated points."""
        Ra, ta = self.poses[a]
        Rb, tb = self.poses[b]
        Ka, Kb = self.dataset.views[a].K, self.dataset.views[b].K
        keep = (depths(X, Ra, ta) > 0) & (depths(X, Rb, tb) > 0)
        keep &= reprojection_errors(X, xy_a, Ka, Ra, ta) < MAX_REPROJ_ERROR_PX
        keep &= reprojection_errors(X, xy_b, Kb, Rb, tb) < MAX_REPROJ_ERROR_PX
        angles = triangulation_angles(X, camera_center(Ra, ta), camera_center(Rb, tb))
        keep &= angles > MIN_TRIANGULATION_ANGLE_DEG
        return keep

    # ------------------------------------------------------------------- growth

    def correspondences_2d3d(
        self, view_idx: int
    ) -> tuple[list[int], np.ndarray, np.ndarray, list[int]]:
        """2D keypoints of an unregistered view already linked to 3D points.

        A keypoint in the new view matched to a registered view's keypoint that
        owns a 3D point gives one 2D↔3D pair — PnP fuel.

        Returns (keypoint indices, (N,2) pixels, (N,3) world points, point ids).
        """
        seen: dict[int, int] = {}  # kp in new view -> point id (first wins)
        for r in self.registration_order:
            for kp_new, kp_r in self.matches_between(view_idx, r):
                pid = self.point_for_kp.get((r, int(kp_r)))
                if pid is not None and int(kp_new) not in seen:
                    seen[int(kp_new)] = pid
        kps = list(seen)
        pts2d = self.dataset.views[view_idx].kp_xy[kps]
        pts3d = np.array([self.points[seen[k]].xyz for k in kps]).reshape(-1, 3)
        return kps, pts2d, pts3d, [seen[k] for k in kps]

    def select_next_view(self, exclude: set[int] | None = None) -> int | None:
        """Unregistered view with the most 2D↔3D links (COLMAP's core heuristic)."""
        exclude = exclude or set()
        unreg = [i for i in range(len(self.dataset)) if i not in self.poses and i not in exclude]
        best, best_n = None, 0
        for i in unreg:
            n = len(self.correspondences_2d3d(i)[0])
            if n > best_n:
                best, best_n = i, n
        return best if best_n >= 12 else None

    def register_view(self, view_idx: int) -> bool:
        view = self.dataset.views[view_idx]
        kps, pts2d, pts3d, pids = self.correspondences_2d3d(view_idx)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            view.K,
            None,
            reprojectionError=PNP_RANSAC_THRESHOLD_PX,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 12:
            logger.warning(
                f"{view.name}: PnP failed ({0 if inliers is None else len(inliers)} inliers)"
            )
            return False
        R, _ = cv2.Rodrigues(rvec)
        self.poses[view_idx] = (R, tvec.reshape(3))
        self.registration_order.append(view_idx)

        # Record the inlier observations on their existing points.
        for li in inliers.ravel():
            kp_idx, pid = kps[li], pids[li]
            if (view_idx, kp_idx) not in self.point_for_kp and view_idx not in self.points[pid].obs:
                self.points[pid].obs[view_idx] = kp_idx
                self.point_for_kp[(view_idx, kp_idx)] = pid
        logger.info(
            f"registered {view.name}: {len(inliers)}/{len(pts2d)} PnP inliers, "
            f"{len(self.poses)}/{len(self.dataset)} views in"
        )
        self._triangulate_new(view_idx)
        return True

    def _triangulate_new(self, view_idx: int) -> None:
        """Create points from matches between the new view and every registered view."""
        va = self.dataset.views[view_idx]
        n_new = 0
        for r in self.registration_order:
            if r == view_idx:
                continue
            m = self.matches_between(view_idx, r)
            if len(m) == 0:
                continue
            # Only matches where NEITHER end already has a point.
            fresh = np.array(
                [
                    (kn, kr)
                    for kn, kr in m
                    if (view_idx, int(kn)) not in self.point_for_kp
                    and (r, int(kr)) not in self.point_for_kp
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            if len(fresh) == 0:
                continue
            vb = self.dataset.views[r]
            xy_a, xy_b = va.kp_xy[fresh[:, 0]], vb.kp_xy[fresh[:, 1]]
            X = triangulate(
                projection_matrix(va.K, *self.poses[view_idx]),
                projection_matrix(vb.K, *self.poses[r]),
                xy_a,
                xy_b,
            )
            keep = self._triangulation_filter(X, view_idx, r, xy_a, xy_b)
            for k in np.flatnonzero(keep):
                self._add_point(
                    X[k], {view_idx: int(fresh[k, 0]), r: int(fresh[k, 1])}
                )
            n_new += int(keep.sum())
        logger.debug(f"{va.name}: +{n_new} new points ({len(self.points)} total)")

    # ------------------------------------------------------------------ quality

    def observation_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten every (point, view) observation → (view_idx, point_id, xy)."""
        cams, pids, xys = [], [], []
        for pid, pt in self.points.items():
            for v, kp in pt.obs.items():
                cams.append(v)
                pids.append(pid)
                xys.append(self.dataset.views[v].keypoints[kp].pt)
        return np.array(cams), np.array(pids), np.array(xys, dtype=np.float64)

    def reprojection_error_stats(self) -> dict[str, float]:
        """Vectorized mean/median reprojection error over every observation."""
        cams, pids, xys = self.observation_arrays()
        pid_to_row = {p: i for i, p in enumerate(sorted(self.points))}
        X = np.array([self.points[p].xyz for p in sorted(self.points)]).reshape(-1, 3)
        R_all = np.zeros((len(self.dataset), 3, 3))
        t_all = np.zeros((len(self.dataset), 3))
        K_all = np.stack([v.K for v in self.dataset.views])
        for v, (R, t) in self.poses.items():
            R_all[v], t_all[v] = R, t
        rows = np.array([pid_to_row[p] for p in pids])
        Xc = np.einsum("oij,oj->oi", R_all[cams], X[rows]) + t_all[cams]
        uvw = np.einsum("oij,oj->oi", K_all[cams], Xc)
        errs = np.linalg.norm(uvw[:, :2] / uvw[:, 2:3] - xys, axis=1)
        return {
            "n_points": int(len(self.points)),
            "n_obs": int(len(errs)),
            "mean_px": float(errs.mean()),
            "median_px": float(np.median(errs)),
        }

    def filter_points(self, max_error_px: float = MAX_REPROJ_ERROR_PX) -> int:
        """Drop observations reprojecting badly; drop points left with <2 views."""
        removed = 0
        for pid in list(self.points):
            pt = self.points[pid]
            for v in list(pt.obs):
                R, t = self.poses[v]
                xy = np.array(self.dataset.views[v].keypoints[pt.obs[v]].pt).reshape(1, 2)
                err = reprojection_errors(
                    pt.xyz.reshape(1, 3), xy, self.dataset.views[v].K, R, t
                )[0]
                if err > max_error_px:
                    self.point_for_kp.pop((v, pt.obs[v]), None)
                    del pt.obs[v]
            if len(pt.obs) < 2:
                self._remove_point(pid)
                removed += 1
        return removed

    # ---------------------------------------------------------------- snapshots

    def snapshot(self) -> dict:
        """Poses + colored points as plain arrays (for saving/replay/Rerun)."""
        pids = sorted(self.points)
        return {
            "view_indices": np.array(self.registration_order, dtype=np.int64),
            "R": np.stack([self.poses[i][0] for i in self.registration_order]),
            "t": np.stack([self.poses[i][1] for i in self.registration_order]),
            "xyz": np.array([self.points[p].xyz for p in pids]).reshape(-1, 3),
            "rgb": np.array([self.points[p].color for p in pids], dtype=np.uint8).reshape(-1, 3),
        }
