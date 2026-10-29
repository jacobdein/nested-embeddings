import pytest

from pyramid_sim import toy


def test_toy_distances_and_ranks():
    d = toy.distances_to()
    expect_raw = {"B": 0.16, "C": 1.14, "E": 1.90, "D": 2.10}
    expect_diff = {"B": 0.26, "E": 1.21, "C": 1.51, "D": 2.45}
    for name, expect in (("raw", expect_raw), ("diff", expect_diff)):
        got = {k[0]: x for k, x in d[name].items()}
        for k, x in expect.items():
            assert got[k] == pytest.approx(x, abs=0.005), (name, k)
        assert sorted(got, key=got.get) == list(expect), name  # same rank order


def test_toy_level2_contrast():
    got = {k[0]: x for k, x in toy.level2_dot().items()}
    expect = {"B": 0.90, "C": 0.08, "E": -0.15, "D": -1.20}
    for k, x in expect.items():
        assert got[k] == pytest.approx(x, abs=0.005), k
    assert sorted(got, key=got.get, reverse=True) == list(expect)
