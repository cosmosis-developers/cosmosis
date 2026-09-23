import numpy as np
import pytest

from cosmosis.postprocessing.statistics import _finite_standard_deviations


def test_finite_standard_deviations_accepts_valid_covariance():
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    np.testing.assert_allclose(_finite_standard_deviations(covariance), [2.0, 3.0])


@pytest.mark.parametrize("covariance", [np.array([[np.nan]]), np.array([[-1.0]])])
def test_finite_standard_deviations_rejects_invalid_diagonal(covariance):
    with pytest.raises(ValueError):
        _finite_standard_deviations(covariance)
