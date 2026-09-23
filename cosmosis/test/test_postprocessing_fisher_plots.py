import numpy as np

from cosmosis.postprocessing.plots import _fisher_to_plot_covariance


def test_fisher_plot_covariance_matches_reference():
    fisher = np.array([[2.0, 0.3], [0.3, 1.2]])
    np.testing.assert_allclose(_fisher_to_plot_covariance(fisher), np.linalg.inv(fisher))


def test_fisher_plot_covariance_rejects_invalid_matrix():
    with np.testing.assert_raises(ValueError):
        _fisher_to_plot_covariance(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_fisher_plot_covariance_rejects_non_positive_definite_matrix():
    with np.testing.assert_raises(np.linalg.LinAlgError):
        _fisher_to_plot_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
