"""Brute-force checks of the dot-product shortcuts in methods.py."""

import numpy as np
import pytest

from pyramid_sim.methods import LevelStats, similarity_maps
from pyramid_sim.pyramid import Pyramid, up


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(2)
    base = (up(rng.normal(size=(8, 8, 4)), 64) + rng.normal(size=(512, 512, 4))).astype(np.float32)
    p = Pyramid(base, n_levels=10)
    st = LevelStats(p)
    return p, st, similarity_maps(p, st, 100, 300)


def _stack(p, r, c, kind):
    m = [p.mean(k)[r >> (k - 1), c >> (k - 1)].astype(np.float64) for k in range(1, 11)]
    if kind == "means":
        return np.concatenate(m)
    d = [m[i] - m[i + 1] for i in range(9)] + [m[9] - p.gmean]
    return np.concatenate(d)


@pytest.mark.parametrize("x", [(0, 0), (100, 301), (511, 7), (250, 250)])
def test_stack_distances(setup, x):
    p, _, (maps, _, _) = setup
    for kind, key in (("means", "M2 means, L2"), ("diffs", "M3 diffs, L2")):
        want = np.linalg.norm(_stack(p, *x, kind) - _stack(p, 100, 300, kind))
        assert -maps[key][x] == pytest.approx(want, rel=1e-4, abs=1e-3)


@pytest.mark.parametrize("x", [(0, 0), (100, 301), (511, 7)])
def test_contrast_dots(setup, x):
    p, _, (maps, sens, info) = setup
    d = lambda r, c: _stack(p, r, c, "diffs").reshape(10, -1)
    m = lambda r, c: _stack(p, r, c, "means").reshape(10, -1)
    assert maps["M3 slot 6 dot"][x] == pytest.approx(d(*x)[5] @ d(100, 300)[5], rel=1e-4, abs=1e-4)
    band = lambda r, c: m(r, c)[4] - m(r, c)[7]
    assert maps["M3 band 5-8 dot"][x] == pytest.approx(band(*x) @ band(100, 300), rel=1e-4, abs=1e-4)
    assert sens["slot 1 dot"][x] == pytest.approx(d(*x)[0] @ d(100, 300)[0], rel=1e-4, abs=1e-4)
    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    assert maps["M1 level 8 cos"][x] == pytest.approx(cos(m(*x)[7], m(100, 300)[7]), abs=1e-5)
