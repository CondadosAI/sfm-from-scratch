"""SIFT features, ratio-test matching, and the geometrically verified pair graph."""

import itertools
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from sfm_from_scratch.config import (
    LOWE_RATIO,
    MIN_PAIR_MATCHES,
    RANSAC_THRESHOLD_PX,
    SIFT_NFEATURES,
)
from sfm_from_scratch.core.dataset import Dataset


def extract_features(dataset: Dataset) -> None:
    """Detect SIFT keypoints + descriptors for every view, in place."""
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    for view in dataset.views:
        view.keypoints, view.descriptors = sift.detectAndCompute(view.load_gray(), None)
        logger.debug(f"{view.name}: {len(view.keypoints)} keypoints")
    counts = [len(v.keypoints) for v in dataset.views]
    logger.info(
        f"SIFT: {int(np.mean(counts))} keypoints/image on average "
        f"(min {min(counts)}, max {max(counts)})"
    )


def match_pair(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Lowe ratio-test matches between two descriptor sets → (N,2) index pairs.

    knnMatch returns the 2 nearest neighbours; a match is kept only if the best
    is clearly better than the runner-up — ambiguous matches (repetitive
    texture) fail this test.
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in knn if m.distance < LOWE_RATIO * n.distance]
    return np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int64).reshape(-1, 2)


def verify_pair(
    xy_a: np.ndarray, xy_b: np.ndarray, matches: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Keep only matches consistent with a single essential matrix (RANSAC).

    The ratio test kills ambiguous descriptors; this kills matches that are
    photometrically plausible but geometrically impossible for ANY rigid
    two-view configuration.
    """
    if len(matches) < MIN_PAIR_MATCHES:
        return matches[:0]
    pts_a = xy_a[matches[:, 0]]
    pts_b = xy_b[matches[:, 1]]
    _, inlier_mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=RANSAC_THRESHOLD_PX
    )
    if inlier_mask is None:
        return matches[:0]
    return matches[inlier_mask.ravel().astype(bool)]


def build_match_graph(
    dataset: Dataset, cache_path: Path | None = None
) -> dict[tuple[int, int], np.ndarray]:
    """Exhaustively match and verify every image pair.

    Returns {(i, j): (N,2) keypoint-index pairs}, i < j, verified pairs only.
    47 images = 1081 pairs — brute force is fine at this scale and needs no
    ordering assumptions (COLMAP offers smarter schemes for thousands of images).
    """
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        graph = {tuple(map(int, k.split("_"))): data[k] for k in data.files}
        logger.info(f"match graph: {len(graph)} verified pairs loaded from {cache_path.name}")
        return graph

    xy = [v.kp_xy for v in dataset.views]
    graph: dict[tuple[int, int], np.ndarray] = {}
    n_pairs = 0
    for i, j in itertools.combinations(range(len(dataset)), 2):
        n_pairs += 1
        raw = match_pair(dataset.views[i].descriptors, dataset.views[j].descriptors)
        inliers = verify_pair(xy[i], xy[j], raw, dataset.views[i].K)
        if len(inliers) >= MIN_PAIR_MATCHES:
            graph[(i, j)] = inliers
    logger.info(f"match graph: {len(graph)}/{n_pairs} pairs verified")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{f"{i}_{j}": m for (i, j), m in graph.items()})
        logger.info(f"match graph cached to {cache_path}")
    return graph
