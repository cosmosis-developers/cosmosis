from cosmosis.runtime import prior as prior_module
import numpy as np
import math

def check_denorm_norm(p):
    for i in range(10):
        y = np.random.uniform(0, 1)
        x = p.denormalize_from_prior(y)
        yy = p.normalize_to_prior(x)
        print(p, y, x, yy)
        assert math.isclose(y, yy)

def test_uniform():
    p = prior_module.UniformPrior(3.0, 7.0)
    check_denorm_norm(p)

def test_gaussian():
    p = prior_module.GaussianPrior(-2.0, -0.1)
    check_denorm_norm(p)

def test_truncated_gaussian():
    p = prior_module.TruncatedGaussianPrior(-1.0, 2.0, -3.0, 2.0)
    check_denorm_norm(p)

def test_exponential():
    p = prior_module.ExponentialPrior(2.2)
    check_denorm_norm(p)

def test_truncated_exponential():
    p = prior_module.TruncatedExponentialPrior(2.2, 0.1, 1.2)
    check_denorm_norm(p)

def test_truncated_one_over_x():
    p = prior_module.TruncatedOneoverxPrior(2.0, 5.0)
    check_denorm_norm(p)
