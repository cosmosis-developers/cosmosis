import io
from pathlib import Path

from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline


def test_automatic_boundary_limits_cache_context():
    slow = Path('/tmp/cache_scope_slow.py')
    fast = Path('/tmp/cache_scope_fast.py')
    slow.write_text(
        "def setup(options): return {}\n"
        "def execute(block, config):\n"
        "    block['likelihoods', 'slow'] = -0.5 * block['parameters', 'p1']**2\n"
        "    return 0\n"
    )
    fast.write_text(
        "def setup(options): return {}\n"
        "def execute(block, config):\n"
        "    block['likelihoods', 'fast'] = -0.5 * block['parameters', 'p2']**2\n"
        "    return 0\n"
    )
    ini = Inifile(io.StringIO(f'''
[pipeline]
modules = slow fast
values = cosmosis/test/example-values.ini
priors = cosmosis/test/example-priors.ini
quiet = T
fast_slow = T
[slow]
file = {slow}
[fast]
file = {fast}
'''))
    pipeline = LikelihoodPipeline(ini)
    pipeline.setup_fast_subspaces()
    try:
        context = pipeline.slow_subspace_cache.cache_context
        assert pipeline.slow_subspace_cache.split_index == 1
        assert 'slow' in context
        assert 'fast' not in context
    finally:
        pipeline.cleanup()
