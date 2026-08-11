"""Static article figures: keypoints, ratio-test vs RANSAC matches, epipolar lines."""

import click
import cv2
import numpy as np
from loguru import logger

from sfm_from_scratch.config import DATA_DIR, OUTPUT_DIR
from sfm_from_scratch.core.dataset import load_middlebury
from sfm_from_scratch.core.features import extract_features, match_pair, verify_pair
from sfm_from_scratch.core.geometry import estimate_relative_pose


@click.command()
@click.option("--dataset", "name", default="templeRing", show_default=True)
@click.option("--pair", nargs=2, type=int, default=(0, 2), show_default=True)
def figures(name: str, pair: tuple[int, int]) -> None:
    """Write keypoints/matches/epipolar-line overlays to output/figures/."""
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_middlebury(DATA_DIR, name)
    a, b = pair
    dataset.views = [dataset.views[a], dataset.views[b]]
    extract_features(dataset)
    va, vb = dataset.views

    # 1. Keypoints on image A (size = detected scale).
    img_kp = cv2.drawKeypoints(
        va.load_bgr(), va.keypoints, None, color=(53, 230, 163),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(fig_dir / "keypoints.png"), img_kp)

    # 2. Matches: ratio-test survivors vs RANSAC-verified survivors.
    raw = match_pair(va.descriptors, vb.descriptors)
    inl = verify_pair(va.kp_xy, vb.kp_xy, raw, va.K)
    for label, m in [("matches_ratio", raw), ("matches_ransac", inl)]:
        dm = [cv2.DMatch(int(q), int(t), 0.0) for q, t in m]
        img_m = cv2.drawMatches(
            va.load_bgr(), va.keypoints, vb.load_bgr(), vb.keypoints, dm, None,
            matchColor=(53, 230, 163), singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(fig_dir / f"{label}.png"), img_m)
    logger.info(f"pair {va.name},{vb.name}: {len(raw)} ratio matches → {len(inl)} RANSAC inliers")

    # 3. Epipolar lines: F from the estimated relative pose (K known).
    xy_a, xy_b = va.kp_xy[inl[:, 0]], vb.kp_xy[inl[:, 1]]
    R, t, mask = estimate_relative_pose(xy_a, xy_b, va.K)
    Kinv = np.linalg.inv(va.K)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = Kinv.T @ tx @ R @ Kinv
    img_a, img_b = va.load_bgr(), vb.load_bgr()
    h, w = img_b.shape[:2]
    rng = np.random.default_rng(7)
    picks = rng.choice(np.flatnonzero(mask), size=8, replace=False)
    colors = [tuple(int(c) for c in rng.integers(80, 255, 3)) for _ in picks]
    for k, color in zip(picks, colors, strict=True):
        pa, pb = xy_a[k], xy_b[k]
        line = F @ np.array([pa[0], pa[1], 1.0])
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = w, int(-(line[0] * w + line[2]) / line[1])
        cv2.circle(img_a, (int(pa[0]), int(pa[1])), 6, color, 2)
        cv2.line(img_b, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(img_b, (int(pb[0]), int(pb[1])), 6, color, 2)
    cv2.imwrite(str(fig_dir / "epipolar_a.png"), img_a)
    cv2.imwrite(str(fig_dir / "epipolar_b.png"), img_b)
    cv2.imwrite(str(fig_dir / "epipolar.png"), cv2.hconcat([img_a, img_b]))
    logger.info(f"figures written to {fig_dir}")


if __name__ == "__main__":
    figures()
