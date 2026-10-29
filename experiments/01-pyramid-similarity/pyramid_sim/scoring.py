"""WorldCover-based scoring of similarity maps.

Labels are used only here, to score hits, never as features. Window shares are taken
over *centred* square windows (not the dyadic cells), so the scoring grid is not the
pyramid's grid and the pyramid gets no free alignment.
"""

from __future__ import annotations

import numpy as np

# ESA WorldCover v200 class codes
TREE, SHRUB, GRASS, CROP, BUILT, BARE, WATER, WETLAND = 10, 20, 30, 40, 50, 60, 80, 90
GREEN = (TREE, SHRUB, GRASS, CROP, WETLAND)
OPEN = (GRASS, CROP)

THIN_LEVEL = 5  # one hit per 16 x 16 px (160 m) dyadic cell


def box_mean(ind: np.ndarray, size: int) -> np.ndarray:
    """Mean of `ind` over a size x size window centred on each pixel (rows/cols
    [i - size/2, i + size/2)), clipped at the AOI edge (mean over the part inside)."""
    h, w = ind.shape
    s = np.zeros((h + 1, w + 1))
    s[1:, 1:] = ind.astype(np.float64).cumsum(0).cumsum(1)
    half = size // 2
    r0 = np.clip(np.arange(h) - half, 0, h)
    r1 = np.clip(np.arange(h) + size - half, 0, h)
    c0 = np.clip(np.arange(w) - half, 0, w)
    c1 = np.clip(np.arange(w) + size - half, 0, w)
    tot = s[r1][:, c1] - s[r0][:, c1] - s[r1][:, c0] + s[r0][:, c0]
    area = (r1 - r0)[:, None] * (c1 - c0)[None, :]
    return (tot / area).astype(np.float32)


class Shares:
    """Lazily computed centred-window class shares: shares(classes, size)."""

    def __init__(self, wc: np.ndarray):
        self.wc = wc
        self._cache: dict = {}

    def __call__(self, classes, size: int) -> np.ndarray:
        classes = (classes,) if isinstance(classes, int) else tuple(sorted(classes))
        key = (classes, size)
        if key not in self._cache:
            self._cache[key] = box_mean(np.isin(self.wc, classes), size)
        return self._cache[key]

    def is_(self, classes) -> np.ndarray:
        classes = (classes,) if isinstance(classes, int) else classes
        return np.isin(self.wc, classes)


def top_cells(
    score: np.ndarray, k: int, exclude_cell: tuple[int, int] | None, seed: int = 0,
    exclude_radius: int = 0,
):
    """Top-k hits thinned to one per level-5 cell.

    Each cell is represented by its best-scoring pixel; cells are ranked by that score.
    Ties (within a cell, and between cells for coarse single-level methods whose
    score is constant over many cells) are broken at random with a fixed seed, so a
    method gets no credit for an arbitrary pixel order. With `exclude_radius` > 0, every
    cell whose centre lies within that many pixels (Chebyshev) of the query is dropped
    too. Returns (rows, cols) of hits.
    """
    h, w = score.shape
    f = 2 ** (THIN_LEVEL - 1)
    rng = np.random.default_rng(seed)
    flat = score.ravel()
    order = np.lexsort((rng.random(flat.size), -np.nan_to_num(flat, nan=-np.inf)))
    rr, cc = np.divmod(order, w)
    cell = (rr // f) * (w // f) + (cc // f)
    _, first = np.unique(cell, return_index=True)  # first (= best) pixel of each cell
    first.sort()
    pick = order[first]
    pr, pc = np.divmod(pick, w)
    if exclude_cell is not None:
        keep = ~((pr // f == exclude_cell[0] // f) & (pc // f == exclude_cell[1] // f))
        if exclude_radius:
            cr, cc = (pr // f) * f + f / 2, (pc // f) * f + f / 2
            near = np.maximum(np.abs(cr - exclude_cell[0]), np.abs(cc - exclude_cell[1]))
            keep &= near > exclude_radius
        pr, pc = pr[keep], pc[keep]
    return pr[:k], pc[:k]


def precision(match: np.ndarray, rows, cols) -> float:
    return float(match[rows, cols].mean())
