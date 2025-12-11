import numpy as np
import torch
from scipy.ndimage import center_of_mass


# --------------------------------------------
# Region / Quadrant Masks
# --------------------------------------------


def make_region_masks(H=14, W=14):
    y = np.arange(H).reshape(H, 1)
    x = np.arange(W).reshape(1, W)

    regions = {
        "top": (y < H // 3),
        "bottom": (y > 2 * H // 3),
        "left": (x < W // 3),
        "right": (x > 2 * W // 3),
        "center": (
            (y >= H // 3) & (y <= 2 * H // 3) & (x >= W // 3) & (x <= 2 * W // 3)
        ),
    }

    quadrants = {
        "upper_left": (y < H // 2) & (x < W // 2),
        "upper_right": (y < H // 2) & (x >= W // 2),
        "lower_left": (y >= H // 2) & (x < W // 2),
        "lower_right": (y >= H // 2) & (x >= W // 2),
    }

    return regions, quadrants


REGION_MASKS, QUADRANT_MASKS = make_region_masks()


def masked_means(A, masks_dict):
    """Compute mean values for each mask."""
    return {name: float(A[mask].mean()) for name, mask in masks_dict.items()}


# --------------------------------------------
# Per-timestep summary (original + new features)
# --------------------------------------------


def summarize_single_timestep(A, top_k=5):
    H, W = A.shape

    # ---- Center of mass ----
    com = center_of_mass(A)

    # ---- Peak location ----
    flat_idx = np.argmax(A)
    peak = (int(flat_idx // W), int(flat_idx % W))
    peak_value = float(A[peak[0], peak[1]])

    # ---- Spread / entropy ----
    spread = float(np.var(A))

    P = A.flatten() + 1e-12
    P = P / P.sum()
    entropy = float(-np.sum(P * np.log(P)))

    # ---- Bounding box above threshold ----
    thresh = A.mean() + A.std()
    ys, xs = np.where(A >= thresh)
    if len(xs) > 0:
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        bbox = [xmin, ymin, xmax, ymax]
        bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)
        high_area = len(xs)
    else:
        bbox = [0, 0, 0, 0]
        bbox_area = 0
        high_area = 0

    # ---- Top-k peaks ----
    k = min(top_k, A.size)
    top_ix = np.argpartition(A.flatten(), -k)[-k:]
    top_coords = [(int(ix // W), int(ix % W)) for ix in top_ix]

    # ---- Region-level attention ----
    region_means = masked_means(A, REGION_MASKS)

    # ---- Quadrant-level attention ----
    quadrant_means = masked_means(A, QUADRANT_MASKS)

    return {
        "center_of_mass": [float(com[0]), float(com[1])],
        "peak": list(peak),
        "peak_value": peak_value,
        "spread": spread,
        "entropy": entropy,
        "bbox": bbox,
        "bbox_area": int(bbox_area),
        "high_area": int(high_area),
        "top_patches": top_coords,
        "region_means": region_means,
        "quadrant_means": quadrant_means,
    }


# --------------------------------------------
# Temporal statistics
# --------------------------------------------


def aggregate_temporal_features(summaries, attn_raw):
    T = len(summaries)

    # Original trajectories
    coms = np.array([s["center_of_mass"] for s in summaries])
    peaks = np.array([s["peak"] for s in summaries])
    spreads = np.array([s["spread"] for s in summaries])
    entropies = np.array([s["entropy"] for s in summaries])

    # Movement (COM / peak)
    com_diffs = np.sqrt(np.sum(np.diff(coms, axis=0) ** 2, axis=1))
    peak_diffs = np.sqrt(np.sum(np.diff(peaks, axis=0) ** 2, axis=1))

    # Region-level deltas
    region_deltas = []
    for t in range(T - 1):
        inc = []
        dec = []
        r0 = summaries[t]["region_means"]
        r1 = summaries[t + 1]["region_means"]
        for reg in r0.keys():
            if r1[reg] > r0[reg] + 1e-6:
                inc.append(reg)
            elif r1[reg] < r0[reg] - 1e-6:
                dec.append(reg)
        region_deltas.append(
            {
                "timestep": t,
                "increase": inc,
                "decrease": dec,
            }
        )

    # Quadrant-level deltas
    quadrant_deltas = []
    for t in range(T - 1):
        inc = []
        dec = []
        q0 = summaries[t]["quadrant_means"]
        q1 = summaries[t + 1]["quadrant_means"]
        for q in q0.keys():
            if q1[q] > q0[q] + 1e-6:
                inc.append(q)
            elif q1[q] < q0[q] - 1e-6:
                dec.append(q)
        quadrant_deltas.append(
            {
                "timestep": t,
                "increase": inc,
                "decrease": dec,
            }
        )

    return {
        # Original temporal stats
        "com_trajectory": coms.tolist(),
        "peak_trajectory": peaks.tolist(),
        "avg_spread": float(spreads.mean()),
        "avg_entropy": float(entropies.mean()),
        "spread_trend": float(spreads[-1] - spreads[0]),
        "entropy_trend": float(entropies[-1] - entropies[0]),
        "com_total_movement": float(com_diffs.sum()),
        "peak_total_movement": float(peak_diffs.sum()),
        "max_step_com_movement": float(com_diffs.max()),
        "max_step_peak_movement": float(peak_diffs.max()),
        # New temporal structure
        "region_deltas": region_deltas,
        "quadrant_deltas": quadrant_deltas,
    }


# --------------------------------------------
# Full pipeline
# --------------------------------------------


def summarize_attn_tensor(attn, top_k=5):
    attn = attn.detach().cpu().numpy()  # (T, 14, 14)
    T = attn.shape[0]

    timestep_summaries = []
    for t in range(T):
        s = summarize_single_timestep(attn[t], top_k=top_k)
        s["timestep"] = t
        timestep_summaries.append(s)

    temporal_stats = aggregate_temporal_features(timestep_summaries, attn)

    return {
        "timestep_summaries": timestep_summaries,
        "temporal_aggregate": temporal_stats,
    }
