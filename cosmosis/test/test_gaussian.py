from cosmosis.runtime import FunctionModule
from cosmosis.datablock import DataBlock
from cosmosis.gaussian_likelihood import GaussianLikelihood, SingleValueGaussianLikelihood
import numpy as np
import os

def test_gaussian():
    class MyLikelihood(GaussianLikelihood):
        x_section = "aaa"
        x_name = "a"
        y_section = "bbb"
        y_name = "b"
        like_name = "lll"

        def build_data(self):
            x_obs = np.array([1.0, 2.0, 3.0])
            y_obs = x_obs * 2
            return x_obs, y_obs

        def build_covariance(self):
            covmat = np.diag([0.1, 0.1, 0.1])
            return covmat

    mod = MyLikelihood.as_module("my")

    # no extra config info
    mod.setup({"my":{"include_norm":True}})

    block = DataBlock()
    block["aaa", "a"] = np.arange(5.)
    block["bbb", "b"] = np.arange(5.) * 2
    status = mod.execute(block)

    assert status == 0
    assert np.isclose(block["data_vector", "lll_chi2"], 0)
    assert np.isclose(block["data_vector", "lll_log_det"], 3*np.log(0.1))
    assert block["data_vector", "lll_n"] == 3

    assert np.isclose(block["likelihoods", "lll_like"], -3*np.log(0.1)/2)


def test_scalar_covariance_cholesky():
    class MyLikelihood(GaussianLikelihood):
        x_section = "aaa"
        x_name = "a"
        y_section = "bbb"
        y_name = "b"
        like_name = "lll"

        def build_data(self):
            return np.array([1.0]), np.array([2.0])

        def build_covariance(self):
            return np.asarray(0.25)

    mod = MyLikelihood.as_module("my")
    mod.setup({"my":{"include_norm":True}})

    block = DataBlock()
    block["aaa", "a"] = np.arange(5.)
    block["bbb", "b"] = np.arange(5.) + 2.0
    status = mod.execute(block)

    assert status == 0
    assert mod.data.cov.shape == (1, 1)
    assert mod.data.inv_cov.shape == (1, 1)
    assert mod.data.chol.shape == (1, 1)
    assert np.isclose(block["data_vector", "lll_chi2"], 4.0)
    assert np.isclose(block["data_vector", "lll_log_det"], np.log(0.25))
    assert block["data_vector", "lll_n"] == 1


def test_cholesky_chi2_matches_inverse_covariance():
    class MyLikelihood(GaussianLikelihood):
        x_section = "aaa"
        x_name = "a"
        y_section = "bbb"
        y_name = "b"
        like_name = "lll"

        def build_data(self):
            x_obs = np.array([1.0, 2.0])
            y_obs = np.array([1.5, -0.5])
            return x_obs, y_obs

        def build_covariance(self):
            return np.array([[2.0, 0.3], [0.3, 1.0]])

    mod = MyLikelihood.as_module("my")
    mod.setup({"my":{}})
    d = np.array([0.25, -0.75])

    chi2 = mod.data._compute_chi2(d)
    expected = np.einsum("i,ij,j", d, mod.data.inv_cov, d)

    assert np.isclose(chi2, expected)


def test_single_gaussian():

    class MySingleLikelihood(SingleValueGaussianLikelihood):
        section = "sec"
        name = "name"
        like_name = "xxx"
        mean = 3.0
        sigma = 0.1


    mod = MySingleLikelihood.as_module("my2")

    # no extra config info
    mod.setup({"my2":{"include_norm":True, "likelihood_only": False}})

    block = DataBlock()

    block["sec", "name"] = 4.0
    status = mod.execute(block)

    # check cholesky correctly calculated
    assert np.isclose(mod.data.chol, 0.1)

    assert status == 0
    assert np.isclose(block["data_vector", "xxx_chi2"], 100)
    assert np.isclose(block["data_vector", "xxx_log_det"], 2*np.log(0.1))
    assert block["data_vector", "xxx_n"] == 1

    assert np.isclose(block["data_vector", "xxx_theory"], 4.0)
    assert np.isclose(block["data_vector", "xxx_data"], 3.0)
    assert np.isclose(block["data_vector", "xxx_inverse_covariance"], 100.0)
    # sim should be within 10 sigma!
    assert 3.0 < block["data_vector", "xxx_simulation"] < 5.0

    assert np.isclose(block["likelihoods", "xxx_like"], -50.0 - np.log(0.1))


if __name__ == '__main__':
    test_gaussian()

