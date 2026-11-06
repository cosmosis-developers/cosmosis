import numpy as np

from cosmosis.postprocessing.density import _kernel_quadratic_form


def test_kernel_quadratic_form_matches_explicit_reference():
    covariance = np.array([[2.0, 0.3], [0.3, 1.2]])
    window = np.mgrid[-2:3, -1:2]
    result = _kernel_quadratic_form(window, covariance)
    inverse = np.linalg.inv(covariance)
    reference = np.einsum('kij,kl,lij->ij', window, inverse, window)
    np.testing.assert_allclose(result, reference)


def test_kernel_quadratic_form_rejects_singular_covariance():
    with np.testing.assert_raises(np.linalg.LinAlgError):
        _kernel_quadratic_form(np.zeros((2, 1, 1)), np.ones((2, 2)))
