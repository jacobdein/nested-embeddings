"""End to end: fetch -> pyramid -> validation -> queries -> scoring -> figures.

    uv run python -m pyramid_sim.run              # 128 dims, then the 64-d check, then figures
    uv run python -m pyramid_sim.run --dims 128   # one width only
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

from . import config as C
from . import fetch, scoring, toy
from .methods import LevelStats, similarity_maps
from .pyramid import Pyramid, dequantize_hwc
from .queries import EXCLUDE_LEVEL, KS, QUERIES, SLOT
from .scoring import BUILT, OPEN, TREE, WATER, Shares, top_cells

SELF_RADIUS = 64  # px = 640 m, the stricter self-match exclusion
DOM_CLASSES = {"tree": (TREE,), "open": OPEN, "built": (BUILT,), "water": (WATER,)}
FIG_METHODS = [  # the panels / rows shown in figures
    "M0 pixel cos",
    "M1 level 5 cos",
    "M1 level 8 cos",
    "M1 level 10 cos",
    "M2 means, equal w",
    "M2 means, focal+context w",
    "M2 means, pair 5+8",
    "M2 means, L2",
    "M3 diffs, L2",
    f"M3 slot {SLOT} dot",
    "M3 band 5-8 dot",
    "M3 band 5-8 dot, negated",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_stats(base):
    out = {}
    for d in (128, 64):
        n = np.linalg.norm(base[..., :d], axis=-1)
        pct = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        out[f"{d}d"] = dict(zip([f"p{p}" for p in pct], np.percentile(n, pct).round(3).tolist()))
        out[f"{d}d"]["mean"] = round(float(n.mean()), 3)
        if d == base.shape[-1]:
            hist, edges = np.histogram(n, bins=120)
            out["hist"] = {"counts": hist.tolist(), "edges": edges.tolist()}
    return out


def validate(p: Pyramid, base: np.ndarray) -> dict:
    v = {}
    t = time.time()
    v["reconstruction_max_abs_err_top12"] = p.reconstruction_error(12)
    v["reconstruction_max_abs_err_top10"] = p.reconstruction_error(10)
    tss = p.total_ss()
    for top in (12, 10):
        ss = p.ss_by_level(top)
        v[f"ss_by_level_top{top}"] = ss.tolist()
        v[f"closure_rel_err_top{top}"] = abs(ss.sum() - tss) / tss
    v["total_ss"] = tss
    v["n_pixels"] = int(p.count(12).sum())
    log(f"  reconstruction + closure in {time.time() - t:.0f} s")

    # scalevar cross-check on a 256 x 256 sub-AOI (Richmond Park), a few single dims
    import scalevar

    r0, c0 = 1024, 1280
    sub = base[r0 : r0 + 256, c0 : c0 + 256]
    sp = Pyramid(np.ascontiguousarray(sub), n_levels=9)
    dims = [0, 17, 63, 64, 127] if base.shape[-1] == 128 else [0, 17, 40, 63]
    ours = sp.ss_by_level(dims=dims)
    theirs = np.zeros_like(ours)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # numpy input: "no CRS" warning
        for d in dims:
            r = scalevar.scale_variance_raster(
                sub[..., d].astype(np.float64), base_level_factor=2, agg_fun="mean"
            )
            theirs += r.components["sum_squares"].to_numpy()
    v["scalevar"] = {
        "sub_aoi_rows_cols": [r0, c0, 256],
        "dims": dims,
        "ours_ss_by_level": ours.tolist(),
        "scalevar_ss_by_level": theirs.tolist(),
        "max_rel_err": float(np.max(np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-300))),
        "scalevar_version": getattr(scalevar, "__version__", "0.1.1"),
    }
    v["toy"] = toy.summary()
    return v


def dominant(S: Shares, size: int) -> np.ndarray:
    names = list(DOM_CLASSES)
    stack = np.stack([S(cls, size) for cls in DOM_CLASSES.values()])
    return np.array(names)[stack.argmax(0)]


def run(dims: int):
    tag = f"{dims}d"
    out_dir = C.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"fetch ({tag})")
    fetch.fetch_all()
    qattrs = fetch.quantization_attrs()
    power = qattrs["geoemb:quantization"]["power"]
    q = fetch.fetch_embeddings()
    base = dequantize_hwc(q, power, dims=dims)
    del q
    res: dict = {"dims": dims, "quantization": qattrs["geoemb:quantization"]}
    res["nodata_fraction"] = float(np.isnan(base).any(-1).mean())
    res["component_range"] = [float(np.nanmin(base)), float(np.nanmax(base))]
    res["norms"] = norm_stats(base)
    log(f"norms: {res['norms'][tag]}")

    log("pyramid")
    t = time.time()
    p = Pyramid(base, C.N_LEVELS)
    res["pyramid_build_s"] = round(time.time() - t, 1)
    if dims == 128:
        nbytes = p.save(C.PYRAMID_DIR)
        res["pyramid_storage_bytes_float32"] = nbytes
        res["base_bytes_float32"] = int(base.nbytes)

    log("validation")
    res["validation"] = validate(p, base)
    log(f"  closure rel err {res['validation']['closure_rel_err_top12']:.2e}, "
        f"scalevar max rel err {res['validation']['scalevar']['max_rel_err']:.2e}")

    log("level stats")
    st = LevelStats(p)
    # coherence (mean resultant length) per level, for figures
    if dims == 128:
        base_norm = np.linalg.norm(base, axis=-1)
        coh = {}
        for k in (3, 5, 8, 10):
            nm, R = p.coherence(base_norm, k)
            coh[k] = {"norm": nm.astype(np.float32), "R": R.astype(np.float32)}
            res.setdefault("coherence", {})[k] = {
                "R_median": float(np.nanmedian(R)), "R_p5": float(np.nanpercentile(R, 5)),
                "R_p95": float(np.nanpercentile(R, 95)), "norm_median": float(np.nanmedian(nm)),
            }
        np.savez_compressed(C.DATA / "coherence.npz", **{
            f"{n}_{k}": v[n] for k, v in coh.items() for n in v})
        np.savez_compressed(C.DATA / "slot_sq.npz", **{f"k{k}": v.astype(np.float32) for k, v in st.slot_sq.items()})

    wc = fetch.fetch_worldcover()
    S = Shares(wc)
    dom16, dom128 = dominant(S, 16), dominant(S, 128)

    res["queries"] = {}
    for qy in QUERIES:
        log(f"query {qy.key}: {qy.name}")
        match, inverse = qy.match(S), qy.inverse(S)
        assert match[qy.row, qy.col], f"{qy.key} exemplar fails its own definition"
        maps, sens, info = similarity_maps(p, st, qy.row, qy.col)
        f = 2 ** (EXCLUDE_LEVEL - 1)
        qcell = (qy.row, qy.col)
        n_cells_any = lambda m: int(m.reshape(C.N // f, f, C.N // f, f).any((1, 3)).sum())
        qres = {
            "row": qy.row, "col": qy.col,
            "base_rate_match": float(match.mean()), "base_rate_inverse": float(inverse.mean()),
            "cells_with_match": n_cells_any(match), "cells_with_inverse": n_cells_any(inverse),
            "query_dom16": str(dom16[qy.row, qy.col]), "query_dom128": str(dom128[qy.row, qy.col]),
            "query_slot_sq": {k: float(v @ v) for k, v in info["qd"].items()},
            "query_mean_norm": {k: float(np.linalg.norm(v)) for k, v in info["qm"].items()},
            "methods": {}, "sensitivity": {},
        }
        thumbs = {}
        for name, m in maps.items():
            rows, cols = top_cells(m, max(KS), qcell)
            target = inverse if name.endswith("negated") else match
            entry = {f"p@{k}": scoring.precision(target, rows[:k], cols[:k]) for k in KS}
            entry["scored_against"] = "inverse" if name.endswith("negated") else "match"
            # added after the first run: also drop every cell within 640 m of the query,
            # because neighbouring cells of the query's own feature re-matched (Q1, Q3)
            xr, xc = top_cells(m, max(KS), qcell, exclude_radius=SELF_RADIUS)
            for k in KS:
                entry[f"p@{k}_x640"] = scoring.precision(target, xr[:k], xc[:k])
            entry["dom16_same_as_query"] = float((dom16[rows, cols] == qres["query_dom16"]).mean())
            entry["dom128_same_as_query"] = float((dom128[rows, cols] == qres["query_dom128"]).mean())
            pairs = [f"{a}/{b}" for a, b in zip(dom16[rows, cols], dom128[rows, cols])]
            u, cnt = np.unique(pairs, return_counts=True)
            entry["dom16_dom128_top200"] = dict(zip(u.tolist(), cnt.tolist()))
            entry["top_rows"], entry["top_cols"] = rows.tolist(), cols.tolist()
            qres["methods"][name] = entry
            if name in FIG_METHODS:
                thumbs[name] = m.reshape(512, 4, 512, 4).mean((1, 3))
        for name, m in sens.items():
            rows, cols = top_cells(m, max(KS), qcell)
            qres["sensitivity"][name] = {
                f"p@{k}": scoring.precision(match, rows[:k], cols[:k]) for k in KS}
            rows, cols = top_cells(-m, max(KS), qcell)
            qres["sensitivity"][name + ", negated"] = {
                f"p@{k}": scoring.precision(inverse, rows[:k], cols[:k]) for k in KS}

        # do the two layouts rank alike? compare cell-level (thinned) scores
        def cell_best(m):
            return m.reshape(C.N // f, f, C.N // f, f).max((1, 3)).ravel()

        def top_set(name):
            e = qres["methods"][name]
            return set(zip(np.array(e["top_rows"]) // f, np.array(e["top_cols"]) // f))

        def spearman(a, b):
            ra, rb = a.argsort().argsort(), b.argsort().argsort()
            return float(np.corrcoef(ra, rb)[0, 1])

        comp = {}
        for a, b in [("M2 means, L2", "M3 diffs, L2"), ("M2 means, equal w", "M3 diffs, L2"),
                     ("M2 means, pair 5+8", "M3 band 5-8 dot"), ("M0 pixel cos", "M3 diffs, L2")]:
            sa, sb = top_set(a), top_set(b)
            comp[f"{a} vs {b}"] = {
                "top200_jaccard": len(sa & sb) / len(sa | sb),
                "cell_spearman": spearman(cell_best(maps[a]), cell_best(maps[b])),
            }
        qres["layout_agreement"] = comp
        res["queries"][qy.key] = qres
        if dims == 128:
            np.savez_compressed(C.DATA / f"thumbs_{qy.key}.npz", **thumbs)
        p_line = {n: round(e["p@200"], 2) for n, e in qres["methods"].items() if n in FIG_METHODS}
        log(f"  p@200 {p_line}")

    path = out_dir / f"results_{tag}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    log(f"wrote {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 64], choices=(64, 128))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    for d in a.dims:
        run(d)
    if (C.RESULTS / "results_128d.json").exists() and not a.no_figures:
        from . import figures, report

        log("figures")
        figures.make_all()
        report.write_tables()


if __name__ == "__main__":
    main()
