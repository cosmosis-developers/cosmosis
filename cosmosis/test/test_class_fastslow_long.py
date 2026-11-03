import io
import os
import time
from pathlib import Path

from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline


def test_real_class_fastslow_revisit_sequence():
    root = Path('/work/standard-library')
    class_root = root / 'boltzmann/class/class_v3.2.5'
    values = root / 'examples/planck_values.ini'
    generated = Path('/tmp/class_fastslow_values.ini')
    text = values.read_text()
    text = text.replace('a_planck          = 0.9 1.0 1.1', 'a_planck          = 1.0')
    text = text.replace('omch2 = 0.12011', 'omch2 = 0.119 0.12011')
    text = text.replace('mnu = 0.06', 'mnu = 0.06 0.15')
    generated.write_text(text + '\n[nuisance]\nx = -1.0 0.0 1.0\n')
    ini = Inifile(io.StringIO(f'''
[pipeline]
modules = consistency class riess probe
values = {generated}
quiet = T
fast_slow = T
first_fast_module = probe
[consistency]
file = {root}/utility/consistency/consistency_interface.py
cosmomc_theta = T
[class]
file = {root}/boltzmann/class/class_interface.py
version = 3.2.5
cmb = F
lensing = F
mpk = T
nk = 12
zmax = 0.5
[riess]
file = {root}/likelihood/riess16/riess16.py
[probe]
file = /tmp/class_fastslow_probe.py
'''))
    Path('/tmp/class_fastslow_probe.py').write_text(
        "def setup(options): return {}\n"
        "def execute(block, config):\n"
        "    block['likelihoods', 'probe'] = -0.5 * block['nuisance', 'x']**2\n"
        "    return 0\n"
    )
    previous = os.getcwd()
    os.chdir(class_root)
    try:
        pipeline = LikelihoodPipeline(ini)
        pipeline.setup_fast_subspaces()
        starts = []
        original = pipeline.slow_subspace_cache.start_pipeline

        def counted(block):
            start = original(block)
            starts.append(start)
            return start

        pipeline.slow_subspace_cache.start_pipeline = counted
        base = pipeline.start_vector()
        vectors = []
        for omch2_delta, mnu_delta in ((0.0, 0.0), (0.0005, 0.0), (0.0, 0.01), (0.0005, 0.01)):
            vector = base.copy()
            vector[0] += omch2_delta
            vector[1] += mnu_delta
            vectors += [vector, vector.copy()]
        fast_variant = vectors[0].copy()
        fast_variant[-1] += 0.5
        vectors = vectors[:2] + [fast_variant] + vectors[2:]
        vectors += [vectors[0].copy(), vectors[3].copy(), vectors[5].copy(), vectors[7].copy()]
        outputs = [pipeline.run_parameters(vector) for vector in vectors]
        assert all(output is not None for output in outputs)
        assert len(starts) == len(vectors)
        assert sum(start == pipeline.slow_subspace_cache.split_index for start in starts) >= 3
        assert outputs[0]['likelihoods', 'riess16_like'] == outputs[1]['likelihoods', 'riess16_like']
        assert starts[2] == pipeline.slow_subspace_cache.split_index
        assert outputs[2]['likelihoods', 'riess16_like'] == outputs[0]['likelihoods', 'riess16_like']
        assert outputs[2]['likelihoods', 'probe'] != outputs[0]['likelihoods', 'probe']
        assert outputs[3]['likelihoods', 'riess16_like'] == outputs[4]['likelihoods', 'riess16_like']
        assert outputs[5]['likelihoods', 'riess16_like'] == outputs[6]['likelihoods', 'riess16_like']
        assert outputs[7]['likelihoods', 'riess16_like'] == outputs[8]['likelihoods', 'riess16_like']
        assert len(pipeline.slow_subspace_cache.cache) == 3
        assert len({tuple(vector[:2]) for vector in vectors[:8]}) == 4
        print({
            'evaluations': len(vectors),
            'cache_hits': sum(start == pipeline.slow_subspace_cache.split_index for start in starts),
            'cache_misses': sum(start != pipeline.slow_subspace_cache.split_index for start in starts),
        })
        pipeline.cleanup()
    finally:
        os.chdir(previous)
