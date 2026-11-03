import io

from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline


def make_pipeline(alpha, beta):
    ini = Inifile(io.StringIO(f'''
[pipeline]
modules = test1 test2
values = cosmosis/test/example-values.ini
priors = cosmosis/test/example-priors.ini
quiet = T
fast_slow = T
first_fast_module = test2
[test1]
file = cosmosis/test/example_module.py
alpha = {alpha}
[test2]
file = cosmosis/test/example_module.py
beta = {beta}
'''))
    pipeline = LikelihoodPipeline(ini)
    pipeline.setup_fast_subspaces()
    return pipeline


def test_only_slow_module_configuration_is_in_context():
    baseline = make_pipeline(1, 1)
    slow_changed = make_pipeline(2, 1)
    fast_changed = make_pipeline(1, 2)
    try:
        assert baseline.slow_subspace_cache.cache_context != slow_changed.slow_subspace_cache.cache_context
        assert baseline.slow_subspace_cache.cache_context == fast_changed.slow_subspace_cache.cache_context
        assert "test1" in baseline.slow_subspace_cache.cache_context
        assert "test2" not in baseline.slow_subspace_cache.cache_context
    finally:
        baseline.cleanup()
        slow_changed.cleanup()
        fast_changed.cleanup()
