"""PNG figures from results_128d.json and the data/ caches."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from . import config as C  # noqa: E402
from . import fetch  # noqa: E402
from .queries import QUERIES, SLOT  # noqa: E402
from .run import FIG_METHODS  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FAMILY = {"M0": "#898781", "M1": "#1baf7a", "M2": "#2a78d6", "M3": "#eb6834"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
WC_COLORS = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000",
             60: "#b4b4b4", 80: "#0064c8", 90: "#0096a0"}
WC_NAMES = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 50: "built-up",
            60: "bare", 80: "water", 90: "wetland"}
LEVEL_LABEL = {k: (f"{10 * 2 ** (k - 1)} m" if k <= 7 else f"{10 * 2 ** (k - 1) / 1000:g} km")
               for k in range(1, 13)}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


def _res():
    return json.loads((C.RESULTS / "results_128d.json").read_text())


def _rgb(s2):
    return np.clip(s2.astype(np.float32) / 170.0, 0, 1)


def _wc_rgb(wc):
    lut = np.full((256, 3), 1.0)
    for k, h in WC_COLORS.items():
        lut[k] = matplotlib.colors.to_rgb(h)
    return lut[wc]


def _km_axes(ax, n=C.N):
    ticks = np.arange(0, n + 1, 512)
    ax.set_xticks(ticks, [f"{t / 100:g}" for t in ticks])
    ax.set_yticks(ticks, [f"{t / 100:g}" for t in ticks])
    ax.tick_params(labelsize=7, length=2)


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


LABEL_OFFSET = {"Q1": (8, -14), "Q3": (8, 6), "Q4": (-8, -8)}  # Q1 and Q3 are 450 m apart


def fig_aoi(s2, wc):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.8))
    axs[0].imshow(_rgb(s2))
    axs[0].set_title("Sentinel-2 L2A true colour, 2024-05-20 (tile 30UXC)", loc="left")
    axs[1].imshow(_wc_rgb(wc), interpolation="nearest")
    axs[1].set_title("ESA WorldCover 2021 v200 (scoring only)", loc="left")
    for ax in axs:
        _km_axes(ax)
        ax.set_xlabel("km east of AOI origin (UTM 30N E 675 840 m)")
        for q in QUERIES:
            ax.plot(q.col, q.row, "o", ms=9, mfc="none", mec="white", mew=2.2)
            ax.plot(q.col, q.row, "o", ms=9, mfc="none", mec=INK, mew=1.0)
            off = LABEL_OFFSET.get(q.key, (7, -7))
            ax.annotate(q.key, (q.col, q.row), xytext=off, textcoords="offset points",
                        ha="left" if off[0] > 0 else "right",
                        fontsize=10, fontweight="bold", color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    axs[0].set_ylabel("km south of AOI top (UTM 30N N 5 713 920 m)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=WC_COLORS[k], ec="#00000033") for k in WC_COLORS]
    axs[1].legend(handles, [WC_NAMES[k] for k in WC_COLORS], loc="lower left", fontsize=7,
                  ncols=4, framealpha=0.9)
    _save(fig, "fig1_aoi.png")


def fig_norms_coherence(res):
    hist = res["norms"]["hist"]
    coh = np.load(C.DATA / "coherence.npz")
    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    e = np.array(hist["edges"])
    ax.bar(e[:-1], np.array(hist["counts"]) / 1e3, width=np.diff(e), align="edge",
           color=FAMILY["M2"], ec=SURFACE, lw=0.3)
    n = res["norms"]["128d"]
    ax.axvline(n["p50"], color=INK, lw=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(0.02, 0.98, f"median {n['p50']:.2f}\n1-99 %: {n['p1']:.2f}-{n['p99']:.2f}\n"
            "two modes: vegetation ~2.9-3.1,\nbuilt-up ~3.4 (see README)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2)
    ax.set_xlabel("‖e‖, de-quantised 128-d pixel embedding")
    ax.set_ylabel("pixels (thousands)")
    ax.set_title("Per-pixel norm, 4.19 M pixels", loc="left")
    for i, k in enumerate((5, 8, 10)):
        a = fig.add_subplot(gs[i + 1])
        R = coh[f"R_{k}"]
        im = a.imshow(R, cmap=BLUES, vmin=0.8, vmax=1, interpolation="nearest")
        a.set_title(f"coherence ‖v̄‖ / mean‖e‖, level {k} ({LEVEL_LABEL[k]})", loc="left", fontsize=8.5)
        a.set_xticks([]), a.set_yticks([])
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        med = res["coherence"][str(k)]["R_median"]
        a.text(0.02, 0.02, f"median {med:.3f}", transform=a.transAxes, fontsize=7.5,
               bbox=dict(fc="white", ec="none", alpha=0.8))
    _save(fig, "fig2_norms_coherence.png")


def fig_scale_variance(res):
    v = res["validation"]
    ss = np.array(v["ss_by_level_top12"])[:11]
    share = ss / v["total_ss"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(1, 12)
    ax.bar(x, share * 100, width=0.72, color=FAMILY["M2"], ec=SURFACE)
    for xi, s in zip(x, share):
        ax.text(xi, s * 100 + 0.3, f"{s * 100:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{k}\n{LEVEL_LABEL[k]}" for k in x], fontsize=7)
    ax.set_xlabel("slot k: level-k cell (size shown) vs its parent at level k+1")
    ax.set_ylabel("% of total SS")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Scale variance of the 128-d embedding: SS_k = Σ m_c‖v̄_k − v̄_k+1‖², "
                 "share of Σ‖e − ḡ‖²", loc="left", fontsize=9)
    _save(fig, "fig3_scale_variance.png")


def _pct(a):
    flat = a.ravel()
    r = np.empty(flat.size, np.float32)
    r[np.argsort(flat, kind="stable")] = np.linspace(0, 1, flat.size, dtype=np.float32)
    return r.reshape(a.shape)


def fig_maps(res):
    for q in QUERIES:
        th = np.load(C.DATA / f"thumbs_{q.key}.npz")
        qr = res["queries"][q.key]
        fig, axs = plt.subplots(3, 4, figsize=(13, 10.4))
        for ax, name in zip(axs.ravel(), FIG_METHODS):
            ax.imshow(_pct(th[name]), cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
            ax.plot(q.col / 4, q.row / 4, "o", ms=7, mfc="none", mec="white", mew=2)
            ax.plot(q.col / 4, q.row / 4, "o", ms=7, mfc="none", mec="#eb6834", mew=1)
            m = qr["methods"][name]
            tgt = " (vs inverse)" if m["scored_against"] == "inverse" else ""
            ax.set_title(f"{name}\np@50 {m['p@50']:.2f}  p@200 {m['p@200']:.2f}{tgt}",
                         loc="left", fontsize=8.5)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"{q.key} {q.name}: similarity, shown as percentile rank over the AOI "
                     f"(dark = most similar). Query ○ at row {q.row}, col {q.col}. "
                     f"Base rate {qr['base_rate_match'] * 100:.2f} % of pixels.",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        _save(fig, f"fig4_maps_{q.key}.png")


CHIP_METHODS = ["M0 pixel cos", "M2 means, focal+context w", "M2 means, pair 5+8",
                "M3 diffs, L2", "M3 band 5-8 dot", "M3 band 5-8 dot, negated"]


def fig_hits(res, s2, wc, n_hits=7, half=24):
    from .scoring import Shares

    S = Shares(wc)
    pad = np.pad(_rgb(s2), ((half, half), (half, half), (0, 0)), constant_values=1)
    for q in QUERIES:
        qr = res["queries"][q.key]
        match, inverse = q.match(S), q.inverse(S)
        fig, axs = plt.subplots(len(CHIP_METHODS), n_hits + 1,
                                figsize=(1.55 * (n_hits + 1), 1.75 * len(CHIP_METHODS)))
        for i, name in enumerate(CHIP_METHODS):
            m = qr["methods"][name]
            tgt = inverse if m["scored_against"] == "inverse" else match
            pts = [(q.row, q.col)] + list(zip(m["top_rows"], m["top_cols"]))[:n_hits]
            for j, (r, c) in enumerate(pts):
                ax = axs[i, j]
                ax.imshow(pad[r : r + 2 * half, c : c + 2 * half], interpolation="nearest")
                ax.plot(half, half, "+", color="white", ms=9, mew=2)
                ax.plot(half, half, "+", color=INK, ms=7, mew=0.8)
                ax.set_xticks([]), ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(name.replace(", ", ",\n"), fontsize=7.5, rotation=0,
                                  ha="right", va="center", color=INK)
                    ax.set_title("query" if i == 0 else "", fontsize=8)
                    for s in ax.spines.values():
                        s.set_visible(True), s.set_color(INK), s.set_linewidth(1.5)
                else:
                    ok = bool(tgt[r, c])
                    ax.set_title(f"#{j} {'✓ match' if ok else '✗'}", fontsize=7.5,
                                 color=INK if ok else MUTED)
                    for s in ax.spines.values():
                        s.set_visible(ok), s.set_color(FAMILY["M3"]), s.set_linewidth(2)
        fig.suptitle(f"{q.key} {q.name}: top hits (one per 160 m cell), 480 m Sentinel-2 chips. "
                     "Negated rows are scored against the inverse definition.",
                     x=0.01, ha="left", fontsize=9.5)
        fig.tight_layout()
        _save(fig, f"fig5_hits_{q.key}.png")


PREC_METHODS = [
    "M0 pixel cos", "M1 level 5 cos", "M1 level 8 cos", "M1 level 10 cos",
    "M2 means, equal w", "M2 means, focal+context w", "M2 means, pair 5+8", "M2 means, L2",
    "M3 diffs, L2", f"M3 slot {SLOT} dot", "M3 band 5-8 dot",
]


def fig_precision(res):
    fig, axs = plt.subplots(1, len(QUERIES), figsize=(15, 4.6), sharey=True)
    y = np.arange(len(PREC_METHODS))[::-1]
    for ax, q in zip(axs, QUERIES):
        qr = res["queries"][q.key]
        for yi, name in zip(y, PREC_METHODS):
            m = qr["methods"][name]
            col = FAMILY[name[:2]]
            ax.plot([m["p@50"], m["p@200"]], [yi, yi], color=col, lw=1, alpha=0.6)
            ax.plot(m["p@200"], yi, "o", ms=8, color=col, mec=SURFACE, mew=1.5)
            ax.plot(m["p@50"], yi, "o", ms=7, mfc=SURFACE, mec=col, mew=1.5)
        ax.axvline(qr["base_rate_match"], color=INK2, lw=1, ls=":")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{q.key} {q.name}", loc="left", fontsize=8.5, wrap=True)
        ax.set_xlabel("precision")
    axs[0].set_yticks(y, PREC_METHODS, fontsize=8)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="o", ls="", color=INK2, ms=8, label="p@200"),
         Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, ms=7, label="p@50"),
         Line2D([], [], color=INK2, ls=":", label="base rate (random pixel)")]
    h += [Line2D([], [], marker="s", ls="", color=c, ms=8, label=lab) for lab, c in
          [("M0 pixel", FAMILY["M0"]), ("M1 one level", FAMILY["M1"]),
           ("M2 means stack", FAMILY["M2"]), ("M3 differences", FAMILY["M3"])]]
    fig.legend(handles=h, loc="lower center", ncols=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "fig6_precision.png")


def fig_level_sensitivity(res):
    fig, axs = plt.subplots(1, len(QUERIES), figsize=(15, 3.6), sharey=True)
    ks = np.arange(1, 11)
    for ax, q in zip(axs, QUERIES):
        qr = res["queries"][q.key]
        m1 = [qr["methods"]["M0 pixel cos" if k == 1 else f"M1 level {k} cos"]["p@200"] for k in ks]
        sl = [qr["sensitivity"][f"slot {k} dot"]["p@200"] for k in ks]
        ax.plot(ks, m1, "-o", color=FAMILY["M1"], lw=2, ms=6, mec=SURFACE, label="means: cos at level k")
        ax.plot(ks, sl, "-o", color=FAMILY["M3"], lw=2, ms=6, mec=SURFACE, label="differences: dot on slot k")
        ax.axhline(qr["base_rate_match"], color=INK2, lw=1, ls=":")
        ax.set_xticks(ks, [str(k) for k in ks], fontsize=7)
        ax.set_xlabel("level k (cell = 10 m × 2^(k−1))")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{q.key} {q.name}", loc="left", fontsize=8.5)
    axs[0].set_ylabel("precision@200")
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig7_level_sensitivity.png")


def make_all():
    res = _res()
    s2, wc = fetch.fetch_s2(), fetch.fetch_worldcover()
    fig_aoi(s2, wc)
    fig_norms_coherence(res)
    fig_scale_variance(res)
    fig_maps(res)
    fig_hits(res, s2, wc)
    fig_precision(res)
    fig_level_sensitivity(res)


if __name__ == "__main__":
    make_all()
