"""SIFT-article figures and numbers: DoG pyramid, scales, ratio test, invariance.

Everything the "What makes a point matchable" article quotes is computed here and
saved to output/figures/sift/ + output/sift_numbers.json, so every number in the
prose has an artifact.
"""

import json

import click
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from sfm_from_scratch.config import DATA_DIR, LOWE_RATIO, OUTPUT_DIR
from sfm_from_scratch.core.dataset import load_middlebury
from sfm_from_scratch.core.features import match_pair, verify_pair

# OpenCV SIFT defaults (Lowe 2004 §3): initial 2x upsample, sigma0=1.6,
# 3 layers/octave => 6 Gaussians and 5 DoGs per octave.
SIGMA0 = 1.6
N_LAYERS = 3


def gaussian_octaves(gray: np.ndarray) -> list[list[np.ndarray]]:
    """Replicate the SIFT Gaussian pyramid (base image doubled, as in OpenCV)."""
    base = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    base = cv2.GaussianBlur(base, (0, 0), np.sqrt(SIGMA0**2 - (2 * 0.5) ** 2))
    n_octaves = int(round(np.log2(min(base.shape[:2])) - 2)) + 1
    k = 2 ** (1.0 / N_LAYERS)
    sigmas = [SIGMA0 * k**i for i in range(N_LAYERS + 3)]
    octaves = []
    img = base.astype(np.float32)
    for _ in range(n_octaves):
        level = [img]
        for i in range(1, len(sigmas)):
            inc = np.sqrt(sigmas[i] ** 2 - sigmas[i - 1] ** 2)
            level.append(cv2.GaussianBlur(level[-1], (0, 0), inc))
        octaves.append(level)
        nxt = level[N_LAYERS]  # sigma has doubled here
        img = cv2.resize(nxt, (nxt.shape[1] // 2, nxt.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    return octaves


def unpack_octave(kp: cv2.KeyPoint) -> int:
    """Decode OpenCV's packed keypoint.octave field to the octave index (-1 = upsampled)."""
    octave = kp.octave & 255
    return octave - 256 if octave >= 128 else octave


@click.command()
@click.option("--dataset", "name", default="templeRing", show_default=True)
@click.option("--view", "view_idx", default=0, show_default=True)
@click.option("--pair-b", default=2, show_default=True, help="Second view for matching figures.")
@click.option("--sweep-until", default=12, show_default=True, help="Match view 0 vs 1..N for the viewpoint sweep.")
def sift_figures(name: str, view_idx: int, pair_b: int, sweep_until: int) -> None:
    """Write SIFT-article figures + a JSON of every number quoted in the prose."""
    fig_dir = OUTPUT_DIR / "figures" / "sift"
    fig_dir.mkdir(parents=True, exist_ok=True)
    numbers: dict = {"dataset": name, "view": view_idx, "pair_b": pair_b}

    dataset = load_middlebury(DATA_DIR, name)
    va = dataset.views[view_idx]
    gray = va.load_gray()
    h, w = gray.shape[:2]
    numbers["image_size"] = [w, h]

    # ---- 1. Pyramid geometry + DoG montage --------------------------------
    octaves = gaussian_octaves(gray)
    n_oct = len(octaves)
    dogs_per_octave = N_LAYERS + 2
    numbers["pyramid"] = {
        "n_octaves": n_oct,
        "gaussians_per_octave": N_LAYERS + 3,
        "dogs_per_octave": dogs_per_octave,
        "total_dogs": n_oct * dogs_per_octave,
        "extrema_levels_per_octave": N_LAYERS,
        "sigma0": SIGMA0,
        "base_upsampled": True,
    }
    show_oct = [0, 2, 4]
    fig, axes = plt.subplots(len(show_oct), dogs_per_octave, figsize=(12, 5.4))
    for r, oi in enumerate(show_oct):
        gs = octaves[oi]
        for c in range(dogs_per_octave):
            dog = gs[c + 1] - gs[c]
            v = np.percentile(np.abs(dog), 99) + 1e-6
            axes[r, c].imshow(dog, cmap="RdBu_r", vmin=-v, vmax=v)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"DoG {c}", fontsize=9)
        axes[r, 0].set_ylabel(f"octave {oi}\n{gs[0].shape[1]}×{gs[0].shape[0]}", fontsize=8)
    fig.suptitle(f"Difference-of-Gaussians pyramid — {va.name} ({w}×{h})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "dog_pyramid.png", dpi=150)
    plt.close(fig)

    # ---- 2. Detect: counts, scale + octave distribution -------------------
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(gray, None)
    sizes = np.array([kp.size for kp in kps])
    octs = np.array([unpack_octave(kp) for kp in kps])
    numbers["detection"] = {
        "n_keypoints": len(kps),
        "size_px": {
            "min": round(float(sizes.min()), 2),
            "median": round(float(np.median(sizes)), 2),
            "p90": round(float(np.percentile(sizes, 90)), 2),
            "max": round(float(sizes.max()), 2),
        },
        "octave_histogram": {int(o): int((octs == o).sum()) for o in np.unique(octs)},
    }

    # Keypoints drawn at their detected scale, small vs large split.
    bgr = va.load_bgr()
    med = float(np.median(sizes))
    small = [kp for kp in kps if kp.size <= med]
    large = [kp for kp in kps if kp.size > med]
    img = cv2.drawKeypoints(bgr, small, None, color=(200, 200, 80),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img = cv2.drawKeypoints(img, large, None, color=(53, 230, 163),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(fig_dir / "keypoints_scale.png"), img)

    # ---- 3. Orientation histogram for one strong keypoint -----------------
    strong = max(kps, key=lambda kp: kp.response)
    r = int(round(1.5 * strong.size))
    x, y = int(round(strong.pt[0])), int(round(strong.pt[1]))
    patch = gray[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang.ravel(), bins=36, range=(0, 360), weights=mag.ravel())
    numbers["orientation_example"] = {
        "kp_xy": [round(strong.pt[0], 1), round(strong.pt[1], 1)],
        "kp_size_px": round(strong.size, 2),
        "kp_angle_deg": round(strong.angle, 1),
        "hist_36bin": [round(float(v), 1) for v in hist],
        "dominant_bin_deg": int(np.argmax(hist) * 10 + 5),
    }
    cv2.imwrite(str(fig_dir / "orientation_patch.png"),
                cv2.resize(gray[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1],
                           None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))

    # ---- 4. Ratio test with real distances --------------------------------
    vb = dataset.views[pair_b]
    kpb, descb = sift.detectAndCompute(vb.load_gray(), None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc, descb, k=2)
    ratios = np.array([m.distance / n.distance for m, n in knn if n.distance > 0])
    d1 = np.array([m.distance for m, _ in knn])
    # The clearest accepted match and a firmly rejected (ambiguous) one.
    idx_sorted = np.argsort(ratios)
    clear_i, ambig_i = int(idx_sorted[0]), int(idx_sorted[-1])
    ex = {}
    for tag, i in [("clear", clear_i), ("ambiguous", ambig_i)]:
        m, n = knn[i]
        ex[tag] = {"d1": round(m.distance, 1), "d2": round(n.distance, 1),
                   "ratio": round(m.distance / n.distance, 3)}
    survivors = {t: int((ratios < t).sum()) for t in (0.6, 0.7, 0.75, 0.8, 0.9, 1.0)}
    numbers["ratio_test"] = {
        "n_query_kp": len(kps), "n_train_kp": len(kpb),
        "examples": ex, "survivors_by_threshold": survivors,
        "lowe_ratio_used": LOWE_RATIO,
        "median_d1": round(float(np.median(d1)), 1),
    }
    raw = match_pair(desc, descb)
    inl = verify_pair(np.array([k.pt for k in kps]), np.array([k.pt for k in kpb]), raw, va.K)
    numbers["ratio_test"]["ransac_inliers"] = [len(raw), len(inl)]

    # ---- 5. Invariance experiments (known transforms => exact ground truth)
    inv = {}
    rng_center = (w / 2, h / 2)
    for tag, M in [
        ("rot45", cv2.getRotationMatrix2D(rng_center, 45, 1.0)),
        ("scale_half", np.array([[0.5, 0, w * 0.25], [0, 0.5, h * 0.25]])),
        ("dark_60pct", None),
    ]:
        if M is None:
            img_t = np.clip(gray.astype(np.float32) * 0.4, 0, 255).astype(np.uint8)
        else:
            img_t = cv2.warpAffine(gray, M, (w, h))
        kpt, desct = sift.detectAndCompute(img_t, None)
        mt = match_pair(desc, desct)
        if len(mt) == 0:
            inv[tag] = {"matches": 0, "correct": 0}
            continue
        pa = np.array([kps[i].pt for i in mt[:, 0]])
        pb = np.array([kpt[j].pt for j in mt[:, 1]])
        if M is None:
            pred = pa
        else:
            pred = (M @ np.hstack([pa, np.ones((len(pa), 1))]).T).T
        err = np.linalg.norm(pred - pb, axis=1)
        inv[tag] = {
            "n_kp_transformed": len(kpt),
            "matches": len(mt),
            "correct_3px": int((err < 3).sum()),
            "precision": round(float((err < 3).mean()), 3),
        }
    numbers["invariance"] = inv

    # ---- 6. Viewpoint sweep: view 0 vs 1..N (ring, ~360/47 deg per step) ---
    step_deg = 360.0 / len(dataset.views)
    sweep = []
    for j in range(1, sweep_until + 1):
        vj = dataset.views[view_idx + j]
        kpj, descj = sift.detectAndCompute(vj.load_gray(), None)
        mj = match_pair(desc, descj)
        # RANSAC without the pipeline's 30-match floor, so a low count reports
        # its true inliers instead of a gated zero (the RANSAC-post lesson).
        n_inl = 0
        if len(mj) >= 5:
            pa = np.array([kps[i].pt for i in mj[:, 0]])
            pb = np.array([kpj[i].pt for i in mj[:, 1]])
            _, mask = cv2.findEssentialMat(pa, pb, va.K, method=cv2.RANSAC,
                                           prob=0.999, threshold=1.5)
            n_inl = int(mask.sum()) if mask is not None else 0
        sweep.append({"dv": j, "deg": round(j * step_deg, 1),
                      "ratio_matches": len(mj), "inliers": n_inl,
                      "passes_pipeline_gate": len(mj) >= 30})
    numbers["viewpoint_sweep"] = {"step_deg_assumed": round(step_deg, 2), "rows": sweep}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["deg"] for r in sweep], [r["ratio_matches"] for r in sweep],
            "o-", label="ratio-test matches")
    ax.plot([r["deg"] for r in sweep], [r["inliers"] for r in sweep],
            "s-", label="RANSAC inliers")
    ax.set_xlabel("viewpoint separation (degrees, ring geometry)")
    ax.set_ylabel("matches")
    ax.set_title(f"SIFT matches vs viewpoint change — {name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "viewpoint_sweep.png", dpi=150)
    plt.close(fig)

    # ---- 7. PatchLab photo + preset eigenvalues ---------------------------
    # The article's interactive lab embeds this exact image (upright + CLAHE
    # contrast lift + 360x480) and quotes these eigenvalues for its presets.
    up = cv2.rotate(va.load_bgr(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    lab_img = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    lc, ac, bc = cv2.split(lab_img)
    lc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lc)
    up = cv2.cvtColor(cv2.merge([lc, ac, bc]), cv2.COLOR_LAB2BGR)
    up = cv2.resize(up, (360, 480), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(fig_dir / "patchlab_photo.png"), up)
    lab_gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def eig_at(x: int, y: int, half: int = 10) -> tuple[float, float]:
        p = lab_gray[y - half:y + half + 1, x - half:x + half + 1]
        gx = cv2.Sobel(p, cv2.CV_32F, 1, 0)[1:-1, 1:-1]
        gy = cv2.Sobel(p, cv2.CV_32F, 0, 1)[1:-1, 1:-1]
        m = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                      [np.sum(gx * gy), np.sum(gy * gy)]]) / gx.size
        lo, hi = np.linalg.eigvalsh(m)
        return float(hi), float(lo)

    presets = {"background": (50, 60), "column_edge": (277, 240), "base_corner": (150, 320)}
    numbers["patchlab_presets"] = {
        k: {"xy": [x, y], "l1": round(l1, 3), "l2": round(l2, 3)}
        for k, (x, y) in presets.items() for l1, l2 in [eig_at(x, y)]
    }

    out = OUTPUT_DIR / "sift_numbers.json"
    out.write_text(json.dumps(numbers, indent=2))
    logger.info(f"figures in {fig_dir}, numbers in {out}")


if __name__ == "__main__":
    sift_figures()
