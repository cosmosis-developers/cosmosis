import numpy as np
import pytest

from cosmosis.samplers.maxlike.maxlike_sampler import _covariance_from_hessian


def test_covariance_from_hessian_uses_symmetric_spd_solution():
    hessian = np.array([[4.0, 1.0], [1.0, 3.0]])
    expected = np.linalg.inv(hessian)
    covariance = _covariance_from_hessian(hessian)
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_allclose(covariance, covariance.T)


@pytest.mark.parametrize("hessian", [np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([[1.0, 2.0], [2.0, -1.0]])])
def test_covariance_from_hessian_rejects_non_spd_inputs(hessian):
    with pytest.raises(ValueError):
        _covariance_from_hessian(hessian)
