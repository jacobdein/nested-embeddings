"""The 2-D toy from the brief: forest axis, grass axis; levels 10 m, 160 m, 2.5 km."""

from __future__ import annotations

import numpy as np


def v(f: float) -> np.ndarray:
    return np.array([f, 1 - f])  # forest share f -> 2-D mean vector


G = v(0.5)
PLACES = {
    "A island": (1, 0.90, 0.15),
    "B island": (1, 0.80, 0.20),
    "C forest interior": (1, 1.0, 0.95),
    "D clearing in forest": (0, 0.10, 0.90),
    "E open grassland": (0, 0.0, 0.10),
}


def stacks():
    raw = {k: [v(f) for f in fs] for k, fs in PLACES.items()}
    diff = {k: [r[0] - r[1], r[1] - r[2], r[2] - G] for k, r in raw.items()}
    return raw, diff


def distances_to(query: str = "A island"):
    raw, diff = stacks()
    out = {}
    for name, stack in (("raw", raw), ("diff", diff)):
        q = np.concatenate(stack[query])
        out[name] = {
            k: float(np.linalg.norm(np.concatenate(s) - q)) for k, s in stack.items() if k != query
        }
    return out


def level2_dot(query: str = "A island"):
    _, diff = stacks()
    return {k: float(s[1] @ diff[query][1]) for k, s in diff.items() if k != query}


def summary() -> str:
    d, dots = distances_to(), level2_dot()
    lines = []
    for name in ("raw", "diff"):
        ranked = sorted(d[name].items(), key=lambda kv: kv[1])
        lines.append(f"{name:4s} L2 to A: " + ", ".join(f"{k[0]} {x:.2f}" for k, x in ranked))
    ranked = sorted(dots.items(), key=lambda kv: -kv[1])
    lines.append("level-2 diff dot with A: " + ", ".join(f"{k[0]} {x:.2f}" for k, x in ranked))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
