import io
from pathlib import Path

from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline


def build(values):
    Path('/tmp/cache_scope_slow.py').write_text(
        "def setup(options): return {}\n"
        "def execute(block, config):\n"
        "    block['likelihoods', 'slow'] = -0.5 * block['parameters', 'p1']**2\n"
        "    return 0\n"
    )
    Path('/tmp/cache_scope_fast.py').write_text(
        "def setup(options): return {}\n"
        "def execute(block, config):\n"
        "    block['likelihoods', 'fast'] = -0.5 * block['parameters', 'p2']**2\n"
        "    return 0\n"
    )
    ini = Inifile(io.StringIO(f'''
[pipeline]
modules = slow fast
values = {values}
priors = cosmosis/test/example-priors.ini
quiet = T
fast_slow = T
first_fast_module = fast
[slow]
file = /tmp/cache_scope_slow.py
[fast]
file = /tmp/cache_scope_fast.py
'''))
    pipeline = LikelihoodPipeline(ini)
    pipeline.setup_fast_subspaces()
    return pipeline


def test_values_content_changes_context(tmp_path):
    values = tmp_path / 'values.ini'
    values.write_text('[parameters]\np1=-3.0 0.0 3.0\np2=-3.0 0.0 3.0\n')
    first = build(values)
    try:
        first_context = first.slow_subspace_cache.cache_context
    finally:
        first.cleanup()
    values.write_text('[parameters]\np1=-3.0 0.1 3.0\np2=-3.0 0.0 3.0\n')
    second = build(values)
    try:
        second_context = second.slow_subspace_cache.cache_context
    finally:
        second.cleanup()
    assert first_context != second_context
    assert any(key == 'values_sha256' for key, _ in second_context['pipeline'])
