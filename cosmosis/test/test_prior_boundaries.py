import numpy as np
import pytest

from cosmosis.runtime.prior import TruncatedExponentialPrior, TruncatedOneoverxPrior


def test_truncated_priors_accept_valid_bounds():
    assert np.isfinite(TruncatedExponentialPrior(2.0, -1.0, 4.0).norm)
    assert np.isfinite(TruncatedOneoverxPrior(0.0, 4.0).norm)


@pytest.mark.parametrize("args", [(0.0, 0.0, 1.0), (np.inf, 0.0, 1.0), (1.0, 2.0, 1.0)])
def test_truncated_exponential_rejects_invalid_bounds(args):
    with pytest.raises(ValueError):
        TruncatedExponentialPrior(*args)


@pytest.mark.parametrize("args", [(0.0, 0.0), (1.0, 1.0), (1.0, np.inf)])
def test_truncated_oneoverx_rejects_invalid_bounds(args):
    with pytest.raises(ValueError):
        TruncatedOneoverxPrior(*args)
