import numpy as np
import astropy.units as u

from dust_extinction.averages import CT06_MWGC, JWST6927_CMZ


def test_continuum_below_ice_range():
    # the ice excess is zero below 2.40 micron, so the curve is CT06_MWGC
    ext = JWST6927_CMZ()
    ct06 = CT06_MWGC()
    wave = np.arange(1.24, 2.40, 0.05) * u.micron
    np.testing.assert_allclose(ext(wave), ct06(wave), rtol=1e-12)


def test_ice_excess_in_ak_units():
    # (JWST6927_CMZ - CT06_MWGC) / (A(K)/A(V)) is the tabulated ice excess
    ext = JWST6927_CMZ()
    ct06 = CT06_MWGC()
    np.testing.assert_allclose(ext.ak_over_av, 0.10793, rtol=1e-4)
    wave = np.array([3.0, 3.5, 4.27, 4.63]) * u.micron
    excess = (ext(wave) - ct06(wave)) / ext.ak_over_av
    np.testing.assert_allclose(excess, [0.594, 0.405, 0.327, 0.317], atol=0.005)


def test_water_band_not_flat_topped():
    # the water band must rise to a single maximum near 3.0 micron and fall
    # monotonically to the red until the shoulder at 3.5 micron
    ext = JWST6927_CMZ()
    ct06 = CT06_MWGC()
    wave = np.arange(2.7, 3.6, 0.01) * u.micron
    excess = (ext(wave) - ct06(wave)) / ext.ak_over_av
    peak = wave[np.argmax(excess)].value
    assert 2.98 <= peak <= 3.06
    assert excess.max() > 0.55
    red = excess[wave.value >= peak]
    assert np.all(np.diff(red) <= 1e-4)
    # no plateau: the top 1% of the peak is at most 0.05 micron wide
    near_top = wave[excess >= 0.99 * excess.max()]
    assert near_top.max().value - near_top.min().value <= 0.05


def test_smooth_ice_curve():
    # second differences on the 0.01 micron grid stay small (curve is smooth)
    ext = JWST6927_CMZ()
    ct06 = CT06_MWGC()
    wave = np.arange(2.40, 5.2501, 0.01) * u.micron
    excess = (ext(wave) - ct06(wave)) / ext.ak_over_av
    d2 = np.diff(excess, 2)
    assert np.sqrt(np.mean(d2**2)) < 0.02
    assert np.abs(d2).max() < 0.1


def test_uncertainties_positive_and_small():
    ext = JWST6927_CMZ()
    assert ext.obsdata_axav_unc.shape == ext.obsdata_x.shape
    assert np.all(ext.obsdata_axav_unc >= 0.0)
    assert ext.obsdata_axav_unc.max() < 0.02
