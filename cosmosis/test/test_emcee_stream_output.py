"""Real CosmoSIS pipeline + emcee 3 regression for PR #242."""
from pathlib import Path
import tempfile
import numpy as np
import pytest
import cosmosis
from cosmosis import Inifile, LikelihoodPipeline, InMemoryOutput

from cosmosis.samplers.emcee.emcee_sampler import EmceeSampler

def run(stream, seed, extra):
    np.random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        values = Path(tmp) / 'values.ini'
        values.write_text('[parameters]\np1=-3 0 3\np2=-3 0 3\n')
        ini = Inifile(None, override={
            ('runtime', 'root'): str(Path(cosmosis.__file__).parent / 'test'),
            ('runtime', 'verbosity'): 'quiet',
            ('pipeline', 'modules'): 'test1',
            ('pipeline', 'values'): str(values),
            ('pipeline', 'extra_output'): 'parameters/p3' if extra else '',
            ('test1', 'file'): 'example_module.py',
            ('emcee', 'walkers'): '8',
            ('emcee', 'samples'): '12',
            ('emcee', 'nsteps'): '4',
            ('emcee', 'stream_output'): 'T' if stream else 'F',
        })
        pipeline = LikelihoodPipeline(ini)
        output = InMemoryOutput()
        sampler = EmceeSampler(ini, pipeline, output)
        sampler.config()
        states = []
        while not sampler.is_converged():
            sampler.execute()
            states.append((sampler.p0.copy(), sampler.prob0.copy(),
                           np.array(sampler.blob0, copy=True), sampler.num_samples))
        pipeline.cleanup()
        return np.asarray(output.rows), states, output.final_meta

@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('seed', [42, 2026])
def test_real_pipeline_stream_equivalence(seed, extra):
    old, states_old, meta_old = run(False, seed, extra)
    new, states_new, meta_new = run(True, seed, extra)
    assert old.shape[0] == new.shape[0] == 96
    np.testing.assert_array_equal(old, new)
    assert len(states_old) == len(states_new) == 3
    for old_state, new_state in zip(states_old, states_new):
        for a, b in zip(old_state, new_state):
            np.testing.assert_array_equal(a, b)
    assert meta_old == meta_new
    # log posterior = Gaussian log likelihood + two uniform log priors.
    expected = -(new[:, 0] ** 2 + new[:, 1] ** 2) / 2 - 2*np.log(6)
    np.testing.assert_allclose(new[:, -1], expected, rtol=0, atol=1e-14)
    if extra:
        np.testing.assert_array_equal(new[:, 2], new[:, 0] + new[:, 1])
