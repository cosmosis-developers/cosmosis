import io
from pathlib import Path

from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline


def make_pipeline(alpha, beta):
    repo_root = Path(__file__).resolve().parents[2]
    values = repo_root / 'cosmosis' / 'test' / 'example-values.ini'
    priors = repo_root / 'cosmosis' / 'test' / 'example-priors.ini'
    module = repo_root / 'cosmosis' / 'test' / 'example_module.py'
    ini = Inifile(io.StringIO(f'''
[pipeline]
modules = test1 test2
values = {values}
priors = {priors}
quiet = T
fast_slow = T
first_fast_module = test2
[test1]
file = {module}
alpha = {alpha}
[test2]
file = {module}
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
