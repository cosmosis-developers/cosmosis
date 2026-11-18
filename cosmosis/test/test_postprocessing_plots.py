import numpy as np

from cosmosis.postprocessing.plots import _stable_logsumexp


def test_postprocessing_logsumexp_handles_extreme_values():
    values = np.array([1000.0, 999.0, -1000.0])
    result = _stable_logsumexp(values)
    expected = 1000.0 + np.log1p(np.exp(-1.0))
    np.testing.assert_allclose(result, expected)
    assert np.isfinite(result)


def test_postprocessing_logsumexp_empty_group():
    assert _stable_logsumexp(np.array([])) == -np.inf
