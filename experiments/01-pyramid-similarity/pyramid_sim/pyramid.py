"""Nested dyadic pyramid of pixel embeddings, held as means; differences on demand.

Level 1 is the 10 m pixel. A level-k cell covers 2^(k-1) x 2^(k-1) pixels and its value
is the plain count-weighted mean of the valid level-1 vectors under it. Sums and counts
are accumulated upward (a level-k sum is the sum of its four children's sums), which is
the base-pixel sum regrouped, so every level is a mean over base pixels, never a mean
of means. No level is renormalised.

Arrays are channels-last, (rows, cols, dims). Level 1 is float32 (the de-quantised
values themselves); levels >= 2 are float64 so the variance closure holds to ~1e-12.

The differences ("Laplacian") layout for a stack topped at level K is

    slot_k = mean_k - up(mean_{k+1})   for k < K
    slot_K = mean_K - g                (g = grand mean of the AOI)

so mean_k = g + sum_{j>=k} up(slot_j), and the per-cell squared norm of slot_k is the
Moellering-Tobler scale-variance element at level k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STRIP = 128  # base rows per strip for level-1 work (128 x 2048 x 128 x 8 B = 268 MB)


def dequant_lut(power: float, scale: float = 127.5, nodata: int = -128) -> np.ndarray:
    """256-entry float32 lookup indexed by the int8 code viewed as uint8."""
    q = np.arange(256, dtype=np.int16)
    q = np.where(q > 127, q - 256, q).astype(np.float64)
    v = (np.abs(q) / scale) ** power * np.sign(q)
    v[q == nodata] = np.nan
    return v.astype(np.float32)


def dequantize_hwc(q_chw: np.ndarray, power: float, dims: int | None = None) -> np.ndarray:
    """int8 (C, H, W) -> float32 (H, W, C), NaN at nodata."""
    if dims is not None:
        q_chw = q_chw[:dims]
    q_hwc = np.ascontiguousarray(np.moveaxis(q_chw, 0, -1))
    return dequant_lut(power)[q_hwc.view(np.uint8)]


def block_sum(a: np.ndarray, f: int = 2) -> np.ndarray:
    h, w = a.shape[:2]
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).sum(axis=(1, 3))


def up(a: np.ndarray, f: int) -> np.ndarray:
    """Nearest-neighbour upsample of the first two axes by an integer factor."""
    if f == 1:
        return a
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)


def level_rows(k: int, r0: int, r1: int) -> tuple[int, int]:
    """Level-k row range covering base rows [r0, r1)."""
    return r0 >> (k - 1), ((r1 - 1) >> (k - 1)) + 1


def to_base_rows(a: np.ndarray, k: int, r0: int, r1: int, lo: int = 0) -> np.ndarray:
    """Base-resolution rows [r0, r1) of a level-k array whose first row is level row `lo`."""
    idx = (np.arange(r0, r1) >> (k - 1)) - lo
    return np.repeat(a[idx], 2 ** (k - 1), axis=1)


class Pyramid:
    def __init__(self, base: np.ndarray, n_levels: int):
        h, w, d = base.shape
        assert h % 2 ** (n_levels - 1) == 0 and w % 2 ** (n_levels - 1) == 0
        self.n_levels, self.dims = n_levels, d
        valid = ~np.isnan(base).any(axis=-1)
        self.means = [base]
        self.counts = [valid.astype(np.int64)]

        # level 2 straight from the base, in strips, NaNs contributing nothing
        s2 = np.empty((h // 2, w // 2, d), np.float64)
        for r in range(0, h, STRIP):
            blk = base[r : r + STRIP].astype(np.float64)
            blk[~valid[r : r + STRIP]] = 0.0
            s2[r // 2 : (r + STRIP) // 2] = block_sum(blk)
        sums, cnt = s2, block_sum(self.counts[0])
        for _ in range(2, n_levels + 1):
            self.counts.append(cnt)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.means.append(sums / cnt[..., None])  # NaN where cnt == 0
            if len(self.means) < n_levels:
                sums, cnt = block_sum(sums), block_sum(cnt)
        top_sum = (self.means[-1] * self.counts[-1][..., None]).reshape(-1, d)
        self.gmean = np.nansum(top_sum, axis=0) / self.counts[-1].sum()

    # --- accessors ---------------------------------------------------------
    def mean(self, k: int) -> np.ndarray:
        return self.means[k - 1]

    def count(self, k: int) -> np.ndarray:
        return self.counts[k - 1]

    def cell_size(self, k: int) -> int:
        return 2 ** (k - 1)

    def diffs(self, k: int, top: int | None = None, rows: slice | None = None) -> np.ndarray:
        """Difference slot k (level-k resolution) of the stack topped at `top`.

        `rows` restricts to a slice of level-k rows; use it for k = 1, where the full
        slot is 2048 x 2048 x 128 floats.
        """
        top = self.n_levels if top is None else top
        assert 1 <= k <= top <= self.n_levels
        rows = rows or slice(0, self.mean(k).shape[0])
        m = self.mean(k)[rows].astype(np.float64)
        if k == top:
            return m - self.gmean
        p = self.mean(k + 1)[rows.start // 2 : -(-rows.stop // 2)]
        pu = up(p, 2)[rows.start % 2 : rows.start % 2 + m.shape[0]]
        return m - pu

    def vectors_at(self, r: int, c: int, levels) -> dict[int, np.ndarray]:
        """Mean vectors of the cells containing base pixel (r, c)."""
        return {k: self.mean(k)[r >> (k - 1), c >> (k - 1)].astype(np.float64) for k in levels}

    def slots_at(self, r: int, c: int, top: int) -> dict[int, np.ndarray]:
        m = self.vectors_at(r, c, range(1, top + 1))
        return {k: (m[k] - m[k + 1]) if k < top else (m[k] - self.gmean) for k in m}

    # --- validation ---------------------------------------------------------
    def reconstruction_error(self, top: int | None = None) -> float:
        """max |e_i - (g + sum_k up(slot_k))| over valid base pixels."""
        top = self.n_levels if top is None else top
        h, w = self.mean(1).shape[:2]
        worst = 0.0
        for r in range(0, h, STRIP):
            r1 = min(r + STRIP, h)
            acc = np.broadcast_to(self.gmean, (r1 - r, w, self.dims)).copy()
            for k in range(1, top + 1):
                lo, hi = level_rows(k, r, r1)
                acc += to_base_rows(self.diffs(k, top, slice(lo, hi)), k, r, r1, lo)
            e = self.mean(1)[r : r + STRIP]
            ok = self.counts[0][r : r + STRIP].astype(bool)
            worst = max(worst, float(np.abs(acc - e)[ok].max()))
        return worst

    def total_ss(self) -> float:
        h = self.mean(1).shape[0]
        tot = 0.0
        for r in range(0, h, STRIP):
            e = self.mean(1)[r : r + STRIP].astype(np.float64) - self.gmean
            ok = self.counts[0][r : r + STRIP].astype(bool)
            tot += float(np.einsum("ij,ij->", e[ok], e[ok]))
        return tot

    def ss_by_level(self, top: int | None = None, dims=None) -> np.ndarray:
        """SS_k = sum over level-k cells of m_c * ||slot_k(c)||^2, k = 1..top."""
        top = self.n_levels if top is None else top
        dims = slice(None) if dims is None else dims
        out = np.zeros(top)
        for k in range(1, top + 1):
            n = self.mean(k).shape[0]
            step = STRIP
            for r in range(0, n, step):
                dk = self.diffs(k, top, slice(r, r + step))[..., dims]
                m = self.count(k)[r : r + step]
                ok = m > 0
                out[k - 1] += float(np.einsum("i,ij,ij->", m[ok], dk[ok], dk[ok]))
        return out

    # --- coherence ------------------------------------------------------------
    def coherence(self, base_norm: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(||mean_k||, ||mean_k|| / mean_{i in cell} ||e_i||) at level-k resolution.

        The ratio is a mean resultant length: 1 when every pixel in the cell points the
        same way, smaller when the cell mixes different vectors.
        """
        f = self.cell_size(k)
        nm = np.linalg.norm(self.mean(k), axis=-1)
        ok = self.counts[0].astype(bool)
        mean_norm = block_sum(np.where(ok, base_norm, 0.0), f) / np.maximum(self.count(k), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return nm, nm / mean_norm

    # --- storage ----------------------------------------------------------------
    def save(self, path: Path) -> int:
        """Means (float32) + counts per level at native resolution. Returns bytes."""
        path.mkdir(parents=True, exist_ok=True)
        total = 0
        for k in range(1, self.n_levels + 1):
            p = path / f"level_{k:02d}.npz"
            np.savez(p, mean=self.mean(k).astype(np.float32), count=self.count(k).astype(np.int32))
            total += p.stat().st_size
        np.save(path / "grand_mean.npy", self.gmean)
        return total
