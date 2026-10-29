import numpy as np
import pytest
import scalevar

from pyramid_sim.pyramid import Pyramid, dequant_lut, up


@pytest.fixture(scope="module")
def pyr():
    rng = np.random.default_rng(0)
    # spatially structured field so every level carries variance
    coarse = rng.normal(size=(4, 4, 6))
    base = (up(coarse, 16) + 0.5 * rng.normal(size=(64, 64, 6))).astype(np.float32)
    base[3, 5] = np.nan  # nodata pixels
    base[40:44, 10:12] = np.nan
    return Pyramid(base, n_levels=7)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_means_are_count_weighted_base_means(pyr):
    base = pyr.mean(1)
    for k in range(2, 8):
        f = 2 ** (k - 1)
        for (i, j) in [(0, 0), (40 // f, 10 // f), (1 % (64 // f), 0)]:
            blk = base[i * f : (i + 1) * f, j * f : (j + 1) * f].reshape(-1, 6)
            np.testing.assert_allclose(pyr.mean(k)[i, j], np.nanmean(blk, axis=0), rtol=1e-5, atol=1e-6)
            assert pyr.count(k)[i, j] == (~np.isnan(blk).any(1)).sum()


def test_reconstruction(pyr):
    for top in (7, 5):
        assert pyr.reconstruction_error(top) < 1e-5


def test_closure(pyr):
    for top in (7, 4):
        ss = pyr.ss_by_level(top)
        assert ss.sum() == pytest.approx(pyr.total_ss(), rel=1e-12)


def test_matches_scalevar_per_dim():
    rng = np.random.default_rng(1)
    base = (up(rng.normal(size=(8, 8, 3)), 4) + rng.normal(size=(32, 32, 3))).astype(np.float32)
    p = Pyramid(base, n_levels=6)
    ours = p.ss_by_level()
    theirs = np.zeros(6)
    for d in range(3):
        with pytest.warns(UserWarning):
            r = scalevar.scale_variance_raster(base[..., d].astype(np.float64), base_level_factor=2)
        theirs += r.components["sum_squares"].to_numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_dequant_lut():
    lut = dequant_lut(2.0)
    assert np.isnan(lut[128])  # int8 -128 viewed as uint8
    assert lut[127] == pytest.approx((127 / 127.5) ** 2)
    assert lut[255] == pytest.approx(-((1 / 127.5) ** 2))  # int8 -1
    assert lut[0] == 0


def test_top_cells_thinning_and_exclusion():
    from pyramid_sim.scoring import top_cells

    s = np.zeros((128, 128), np.float32)
    s[20, 20], s[21, 21], s[100, 100], s[60, 60] = 5, 4, 3, 2  # (20,20),(21,21) share a cell
    r, c = top_cells(s, 3, exclude_cell=None)
    assert list(zip(r, c)) == [(20, 20), (100, 100), (60, 60)]
    r, c = top_cells(s, 2, exclude_cell=(101, 101))
    assert list(zip(r, c)) == [(20, 20), (60, 60)]
    r, c = top_cells(s, 1, exclude_cell=(30, 30), exclude_radius=20)
    assert (r[0], c[0]) == (100, 100)
