"""Similarity maps for one query pixel, every method on the same levels (1-10).

Everything reduces to per-level dot products with a handful of probe vectors, because
a difference slot is linear in the means:

    slot_k(x) . p = mean_k(x) . p - mean_{k+1}(x) . p        (k < top)
    slot_top(x) . p = mean_top(x) . p - g . p

so one batched matmul per level serves cosine on means, Euclidean on the stacks and the
contrast dot products. Maps are returned at base (10 m) resolution as float32.
"""

from __future__ import annotations

import numpy as np

from .pyramid import STRIP, Pyramid, up
from .queries import BAND, EQUAL_W, FOCAL_CONTEXT_W, LEVELS, PAIR_LEVELS, SLOT

TOP = max(LEVELS)


class LevelStats:
    """Per-level quantities that do not depend on the query."""

    def __init__(self, p: Pyramid):
        self.p = p
        self.norm = {k: np.linalg.norm(p.mean(k), axis=-1).astype(np.float32) for k in LEVELS}
        # ||slot_k||^2 per level-k cell: the scale-variance element d_c of each cell
        self.slot_sq = {}
        for k in LEVELS:
            n = p.mean(k).shape[0]
            out = np.empty(p.mean(k).shape[:2], np.float64)
            for r in range(0, n, STRIP):
                d = p.diffs(k, TOP, slice(r, r + STRIP))
                out[r : r + STRIP] = np.einsum("ijk,ijk->ij", d, d)
            self.slot_sq[k] = out


def _up(a: np.ndarray, k: int) -> np.ndarray:
    return up(a.astype(np.float32), 2 ** (k - 1))


def similarity_maps(p: Pyramid, st: LevelStats, r: int, c: int, sensitivity: bool = True):
    qm = p.vectors_at(r, c, LEVELS)  # mean vectors of the query's cells
    qd = p.slots_at(r, c, TOP)  # difference slots of the query, stack topped at 10
    a, b = BAND
    qband = qm[a] - qm[b]

    probes = [qm[k] for k in LEVELS] + [qd[k] for k in LEVELS] + [qband]
    P = np.stack(probes, axis=1)  # (D, n_probe)
    PM = {k: i for i, k in enumerate(LEVELS)}
    PD = {k: len(LEVELS) + i for i, k in enumerate(LEVELS)}
    PB = 2 * len(LEVELS)

    dots = {}
    for k in LEVELS:
        m = p.mean(k)
        dots[k] = (m @ P.astype(m.dtype)).astype(np.float64 if k > 1 else np.float32)
    g_dot = p.gmean @ P

    def slot_dot(k: int, j: int) -> np.ndarray:
        """slot_k(x) . probe_j at level-k resolution."""
        if k == TOP:
            return dots[k][..., j] - g_dot[j]
        return dots[k][..., j] - up(dots[k + 1][..., j], 2)

    cos = {}
    for k in LEVELS:
        with np.errstate(invalid="ignore", divide="ignore"):
            cos[k] = dots[k][..., PM[k]] / (st.norm[k] * np.linalg.norm(qm[k]))
    cos_b = {k: _up(cos[k], k) for k in LEVELS}

    maps: dict[str, np.ndarray] = {}
    maps["M0 pixel cos"] = cos_b[1]
    for k in LEVELS[1:]:
        maps[f"M1 level {k} cos"] = cos_b[k]
    maps["M2 means, equal w"] = sum(EQUAL_W[k] * cos_b[k] for k in EQUAL_W)
    maps["M2 means, focal+context w"] = sum(FOCAL_CONTEXT_W[k] * cos_b[k] for k in FOCAL_CONTEXT_W)
    pa, pb = PAIR_LEVELS
    maps[f"M2 means, pair {pa}+{pb}"] = 0.5 * (cos_b[pa] + cos_b[pb])

    # Euclidean over the concatenated stacks: ||u - q||^2 = ||u||^2 - 2 u.q + ||q||^2
    l2_means = sum(
        _up(st.norm[k].astype(np.float64) ** 2 - 2 * dots[k][..., PM[k]] + qm[k] @ qm[k], k)
        for k in LEVELS
    )
    maps["M2 means, L2"] = -np.sqrt(np.maximum(l2_means, 0)).astype(np.float32)
    l2_diffs = sum(
        _up(st.slot_sq[k] - 2 * slot_dot(k, PD[k]) + qd[k] @ qd[k], k) for k in LEVELS
    )
    maps["M3 diffs, L2"] = -np.sqrt(np.maximum(l2_diffs, 0)).astype(np.float32)

    s = _up(slot_dot(SLOT, PD[SLOT]), SLOT)
    maps[f"M3 slot {SLOT} dot"] = s
    maps[f"M3 slot {SLOT} dot, negated"] = -s
    band = _up(dots[a][..., PB], a) - _up(dots[b][..., PB], b)
    maps[f"M3 band {a}-{b} dot"] = band
    maps[f"M3 band {a}-{b} dot, negated"] = -band

    sens = {}
    if sensitivity:
        for k in LEVELS:
            sens[f"slot {k} dot"] = _up(slot_dot(k, PD[k]), k)
    return maps, sens, {"qm": qm, "qd": qd}
