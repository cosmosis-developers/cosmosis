import numpy as np

from cosmosis.plotting.grid_plots import GridPlotter


def test_grid_logsumexp_handles_extreme_log_likelihoods():
    values = np.array([1000.0, 999.0, -1000.0])
    result = GridPlotter._logsumexp(values)
    expected = 1000.0 + np.log1p(np.exp(-1.0))
    np.testing.assert_allclose(result, expected)
    assert np.isfinite(result)


def test_grid_logsumexp_empty_group_is_negative_infinity():
    result = GridPlotter._logsumexp(np.array([]))
    assert result == -np.inf
