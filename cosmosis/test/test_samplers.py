from cosmosis import Inifile, Sampler, LikelihoodPipeline, InMemoryOutput, TextColumnOutput
from cosmosis.postprocessing import postprocessor_for_sampler
from cosmosis.runtime import logs
import tempfile
import os
import sys
import pytest
import numpy as np
from astropy.table import Table


minuit_compiled = os.path.exists(Sampler.get_sampler("minuit").libminuit_name)

# our test priors are uniform on [-3, 3] for two
# parameters, so our expected prior is 1/6 for each of them.
EXPECTED_LOG_PRIOR = 2*np.log(1./6)

def run(name, check_prior, check_extra=True, can_postprocess=True, do_truth=False, no_extra=False, pp_extra=True, pp_2d=True, hints_peak=True, hints_cov=True, **options):

    sampler_class = Sampler.registry[name]

    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-3.0  0.0  3.0\n"
        "p2=-3.0  0.0  3.0\n")
    values.flush()

    override = {
        ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
        ('runtime', 'verbosity'): "noisy",
        ("pipeline", "debug"): "F",
        ("pipeline", "modules"): "test1",
        ("pipeline", "extra_output"): "parameters/p3",
        ("pipeline", "values"): values.name,
        ("test1", "file"): "example_module.py",
    }

    for k,v in options.items():
        override[(name,k)] = str(v)

    if no_extra:
        del override[("pipeline", "extra_output")]


    ini = Inifile(None, override=override)

    # Make the pipeline itself
    pipeline = LikelihoodPipeline(ini)

    output = InMemoryOutput()
    sampler = sampler_class(ini, pipeline, output)
    sampler.config()


    while not sampler.is_converged():
        sampler.execute()

    if check_prior:
        pr = np.array(output['prior'])
        # a few samples might be outside the bounds and have zero prior
        assert np.all((pr==EXPECTED_LOG_PRIOR)|(pr==-np.inf))
        # but not all of them
        assert not np.all(pr==-np.inf)

    if hints_peak:
        assert sampler.distribution_hints.has_peak()
        peak = sampler.distribution_hints.get_peak()
        assert peak.shape == (2,)
        idx = output["post"].argmax()
        assert np.isclose(output["parameters--p1"][idx], peak[0])
        assert np.isclose(output["parameters--p2"][idx], peak[1])
        sampler.distribution_hints.del_peak()
        assert not sampler.distribution_hints.has_peak()
    if hints_cov:
        assert sampler.distribution_hints.has_cov()
        cov = sampler.distribution_hints.get_cov()
        assert cov.shape == (2, 2)



    if check_extra and not no_extra:
        p1 = output['parameters--p1']
        p2 = output['parameters--p2']
        p3 = output['PARAMETERS--P3']
        assert np.all((p1+p2==p3)|(np.isnan(p3)))

    if can_postprocess:
        pp_class = postprocessor_for_sampler(name)
        print(pp_class)
        for getdist in [True, False]:
            with tempfile.TemporaryDirectory() as dirname:
                truth_file = values.name if do_truth else None
                pp = pp_class(output, "Chain", 0, outdir=dirname, prefix=name, truth=truth_file, fatal_errors=True, getdist=getdist)
                pp_files = pp.run()
                pp.finalize()
                pp.save()
                for p in pp_files:
                    print(p)
                postprocess_files = ['parameters--p1', 'parameters--p2']
                if pp_2d:
                    postprocess_files.append('2D_parameters--p2_parameters--p1')
                if check_extra and pp_extra and not no_extra:
                    postprocess_files.append('parameters--p3')
                    if pp_2d:
                        postprocess_files += ['2D_parameters--p3_parameters--p2', '2D_parameters--p3_parameters--p1']
                for p in postprocess_files:
                    filename = f"{dirname}{os.path.sep}{name}_{p}.png"
                    print("WANT ", filename)
                    assert filename in pp_files
                    assert os.path.exists(filename)
                for p in os.listdir(dirname):
                    p = os.path.join(dirname, p)
                    if p.endswith(".txt"):
                        Table.read(p, format='ascii.commented_header')
                        print(f"Read file {p} as a table")


    return output


def test_apriori():
    run('apriori', True, can_postprocess=False, hints_cov=False, nsample=100)

def test_dynesty():
    # dynesty does not support extra params
    run('dynesty', False, check_extra=False, nlive=25, sample='unif')

def test_emcee():
    run('emcee', True, walkers=8, samples=25)
    run('emcee', True, walkers=8, samples=25, a=3.0)

def test_truth():
    run('emcee', True, walkers=8, samples=25, do_truth=True)

def test_fisher():
    run('fisher', False, check_extra=False, hints_peak=False)
    run('fisher', False, check_extra=False, hints_peak=False)

def test_fisher_numdifftools():
    try:
        import numdifftools
    except ImportError:
        pytest.skip("numdifftools not installed")
    run('fisher', False, check_extra=False, hints_peak=False, method="numdifftools")

def test_fisher_smoothing():
    try:
        import derivative
    except ImportError:
        pytest.skip("derivative not installed")
    run('fisher', False, check_extra=False, hints_peak=False, method="smoothing")

def test_grid():
    run('grid', True, pp_extra=False, nsample_dimension=10, hints_cov=False)

def test_gridmax():
    run('gridmax', True, can_postprocess=False, max_iterations=1000, hints_cov=False)

def test_latin_hypercube():
    run('latinhypercube', True, pp_extra=False, nsample=10, hints_cov=False)

# def test_kombine():
#     run('kombine')

def test_maxlike_single():
    output = run('maxlike', True, can_postprocess=False, hints_cov=False)
    assert len(output["post"]) == 1

def test_maxlike_alt():
    # alternative sampler, max-post, output_cov
    with tempfile.TemporaryDirectory() as dirname:
        output_ini = os.path.join(dirname, "output.ini")
        output_cov = os.path.join(dirname, "output_cov.txt")
        output_block = os.path.join(dirname, "output_block")
        run('maxlike', True, can_postprocess=False, method="L-BFGS-B", max_posterior=True, output_ini=output_ini, output_covmat=output_cov, output_block=output_block)
        assert os.path.exists(output_cov)
        assert os.path.exists(output_ini)
        assert os.path.isdir(output_block)
        


def test_maxlike_start_prior_repeat(caplog):
    output = run('maxlike', True, can_postprocess=False, repeats=5, start_method="prior", hints_cov=False)
    assert len(output["post"]) == 5
    assert (np.diff(output["like"]) >= 0).all()
    assert "Starting at a random point in the prior" in caplog.text

def test_maxlike_start_unable_to_repeat():
    # error - no start method specified but need one for repeats
    with pytest.raises(ValueError):
        run('maxlike', True, can_postprocess=False, repeats=5)

def test_maxlike_reiterations():
    # re-run the maxlike starting from the end of the previous run
    output = run('maxlike', True, can_postprocess=False, hints_cov=False, reiterations=2)
    assert len(output["post"]) == 1


def test_maxlike_start_no_start_file():
    # error - no start_input specified
    with pytest.raises(ValueError):
        run('maxlike', True, can_postprocess=False, repeats=5, start_method="chain")

def test_maxlike_start_chain_sample_auto(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # This should work, and default to chain-sample because we have set repeats > 1
        run('maxlike', True, can_postprocess=False, repeats=5, start_method="chain", start_input=f.name, hints_cov=False)
        assert "Starting at random sample of points from chain file" in caplog.text

def test_maxlike_start_chain_maxpost_auto(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # This should work, and default to chain-sample because we have set repeats > 1
        run('maxlike', True, can_postprocess=False, repeats=1, start_method="chain", start_input=f.name, hints_cov=False)
        assert "Starting at best posterior point from chain file" in caplog.text

def test_maxlike_start_chain_maxlike_auto(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight like\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # This should work, and default to chain-sample because we have set repeats > 1
        run('maxlike', True, can_postprocess=False, repeats=1, start_method="chain", start_input=f.name, hints_cov=False)
        assert "Starting at best likelihood point from chain file" in caplog.text


def test_maxlike_start_chain_sample_manual(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # This should work, and default to chain-sample because we have set repeats > 1
        run('maxlike', True, can_postprocess=False, repeats=5, start_method="chain-sample", start_input=f.name, hints_cov=False)
        assert "Starting at random sample of points from chain file" in caplog.text

    # Again with log-weights
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  0.0  0.0\n")
        f.write("0.05 0.0  0.7  1.0\n")
        f.write("-0.1 0.2  0.7  2.0\n")
        f.flush()

        # This should work, and default to chain-sample because we have set repeats > 1
        run('maxlike', True, can_postprocess=False, repeats=5, start_method="chain-sample", start_input=f.name, hints_cov=False)
        assert "Starting at random sample of points from chain file" in caplog.text

    # Again with no weights
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 post\n")
        f.write("0.0 0.1  0.0\n")
        f.write("0.05 0.0  1.0\n")
        f.write("-0.1 0.2   2.0\n")
        f.flush()

        # This should work, and default to chain-sample because we have set repeats > 1
        run('maxlike', True, can_postprocess=False, repeats=5, start_method="chain-sample", start_input=f.name, hints_cov=False)
        assert "Starting at random sample of points from chain file" in caplog.text


def test_maxlike_start_chain_fail(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # should fail - no like column
        with pytest.raises(ValueError):
            run('maxlike', True, can_postprocess=False, repeats=1, start_method="chain-maxlike", start_input=f.name, hints_cov=False)


def test_maxlike_start_max_post(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # start from best element of chain - should work because there is a max-post column
        run('maxlike', True, can_postprocess=False, repeats=1, start_method="chain-maxpost", start_input=f.name, hints_cov=False)
        assert "Starting at best posterior point from chain file" in caplog.text

def test_maxlike_start_last(caplog):
    # Check we can start from a chain file
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1 p2 weight post\n")
        f.write("0.0 0.1  1.0  0.0\n")
        f.write("0.05 0.0  2.0  1.0\n")
        f.write("-0.1 0.2  2.0  2.0\n")
        f.flush()

        # start from best element of chain - should work because there is a max-post column
        run('maxlike', True, can_postprocess=False, repeats=1, start_method="chain-last", start_input=f.name, hints_cov=False)
        assert "Starting from last point in file" in caplog.text



def test_maxlike_start_covmat(caplog):
    # Check we can start from a covmat
    with tempfile.NamedTemporaryFile('w') as f:
        f.write("0.1  0.0\n")
        f.write("0.0 0.08\n")
        f.flush()
        run('maxlike', True, can_postprocess=False, repeats=5, start_method="cov", start_input=f.name, hints_cov=False)
        assert "Starting at a random sample of points from the covariance of chain" in caplog.text


def test_start_estimate():
    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-10.0  0.0  10.0\n"
        "p2=-10.0  0.0  10.0\n")
    values.flush()

    override = {
        # ('runtime', 'verbosity'): "noisy",
        ("pipeline", "modules"): "",
        ("pipeline", "values"): values.name,
    }

    # mock objects just to stop the init breaking
    ini = Inifile(None, override=override)

    pipeline = LikelihoodPipeline(ini)
    sampler = Sampler(ini, pipeline)

    with tempfile.NamedTemporaryFile('w') as f:
        f.write("#p1    p2  weight like post\n")
        f.write(" 1.0   1.0  1.0  1.0   0.0\n") # non-zero weight
        f.write(" 2.0   2.0  0.0  2.0   0.0\n") # max-like
        f.write(" 3.0   3.0  0.0  1.0   1.0\n")
        f.write(" 4.0   4.0  0.0  0.0   2.0\n") # max-post
        f.write(" 5.0   5.0  0.0  0.0   0.0\n") # last
        f.flush()

        # only one sample has any weight so this one should be selected
        p = sampler.start_estimate(method="chain-sample", input_source=f.name, prefer_random=False)
        assert np.allclose(p, [1.0, 1.0])

        # only one sample has any weight so this one should be selected
        p = sampler.start_estimate(method="chain-maxlike", input_source=f.name, prefer_random=False)
        assert np.allclose(p, [2.0, 2.0])

        # last sample should be selected
        p = sampler.start_estimate(method="chain-maxpost", input_source=f.name, prefer_random=False)
        assert np.allclose(p, [4.0, 4.0])

        # last sample should be selected
        p = sampler.start_estimate(method="chain-last", input_source=f.name, prefer_random=False)
        assert np.allclose(p, [5.0, 5.0])

        # fall back to ini file choice
        p = sampler.start_estimate(method="", prefer_random=False)
        assert np.allclose(p, [0.0, 0.0])





def test_bobyqa():
    with tempfile.TemporaryDirectory() as dirname:
        output_cov = os.path.join(dirname, "output_cov.txt")
        run('maxlike', True, can_postprocess=False, method='bobyqa', output_covmat=output_cov)
        assert os.path.exists(output_cov)

        with pytest.raises(ValueError, match="The tolerance for bobyqa must be less than 0.1"):
            run('maxlike', True, can_postprocess=False, method='bobyqa', tolerance=0.2)

def test_metropolis():
    run('metropolis', True, samples=20)
    run('metropolis', True, samples=20, covmat_sample_start=True)

@pytest.mark.skipif(not minuit_compiled,reason="requires Minuit2")
def test_minuit():
    run('minuit', True, can_postprocess=False, hints_cov=False)

def test_multinest():
    run('multinest', True, max_iterations=10000, live_points=50, feedback=False)

def test_pmaxlike():
    run('pmaxlike', True, can_postprocess=False, hints_cov=False)

def test_pmc():
    old_settings = np.seterr(invalid='ignore', divide='ignore')
    run('pmc', True, iterations=3, hints_cov=False)
    np.seterr(**old_settings)  

def test_zeus():
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, verbose=True)
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, moves="differential:2.0  global", tolerance=0.1, patience=5000)

def test_polychord():
    with tempfile.TemporaryDirectory() as base_dir:
        run('polychord', True, live_points=20, feedback=0, base_dir=base_dir, polychord_outfile_root='pc')

def test_snake():
        run('snake', True, pp_extra=False, hints_cov=False)


# Skip in a specific combination which causes a crash I can't track down.
# I think it's a memory thing as it's only when when in the full test suite
# not when run standalone.
@pytest.mark.skipif(os.environ.get("SKIP_NAUTILUS", "0")=="1", reason="nautilus runs out of memory on github actions")
def test_nautilus():
    run('nautilus', True, n_live=200, enlarge_per_dim=1.05,
        split_threshold=95., n_networks=2, n_batch=50, verbose=True, f_live=0.02, n_shell=50)


def test_star():
        run('star', False, pp_extra=False, pp_2d=False, hints_cov=False)

def test_test():
    run('test', False, can_postprocess=False, hints_peak=False, hints_cov=False)

def test_list_sampler():
    # test that the burn and thin parameters work
    # make a mock output file with some samples
    with tempfile.TemporaryDirectory() as dirname:

        input_chain_filename = os.path.join(dirname, "input_chain.txt")
        input_chain = TextColumnOutput(input_chain_filename)
        input_chain.add_column('parameters--p1', float)
        input_chain.add_column('parameters--p2', float)
        nrow = 100
        data = np.random.normal(size=(nrow, 2)).clip(-2.99, 2.99)
        for i in range(nrow):
            input_chain.parameters(data[i, 0], data[i, 1])
        input_chain.close()

        burn = 0.2
        thin = 2

        data = data[20::thin]

        output = run("list", True, can_postprocess=False, filename=input_chain_filename, burn=burn, thin=thin, hints_cov=False)
        p1 = output['parameters--p1']
        p2 = output['parameters--p2']
        p3 = p1 + p2
        expected_like = -(p1**2 + p2**2)/2.
        assert p1.size == 40
        assert np.allclose(p1, data[:, 0])
        assert np.allclose(p2, data[:, 1])
        assert np.allclose(output['PARAMETERS--P3'], p3)
        assert np.allclose(output["post"] - output["prior"], expected_like)

def test_importance_sampler():

    for add_to_likelihood in [True, False]:
        with tempfile.TemporaryDirectory() as dirname:

            input_chain_filename = os.path.join(dirname, "input_chain.txt")
            input_chain = TextColumnOutput(input_chain_filename)
            input_chain.add_column('parameters--p1', float)
            input_chain.add_column('parameters--p2', float)
            input_chain.add_column('post', float)
            nrow = 100
            original_posteriors = -np.arange(nrow).astype(float)
            data = np.random.normal(size=(nrow, 2)).clip(-2.99, 2.99)
            for i in range(nrow):
                input_chain.parameters(data[i, 0], data[i, 1], original_posteriors[i])
            input_chain.close()
            os.system(f"cat {input_chain_filename}")

            output = run("importance", True, can_postprocess=False, input=input_chain_filename, add_to_likelihood=add_to_likelihood, hints_cov=False)
            p1 = output['parameters--p1']
            p2 = output['parameters--p2']
            p3 = p1 + p2
            expected_like = -(p1**2 + p2**2)/2.
            # priors are uniform with range -3 to 3 so normalization should be np.log(1/6)
            expected_prior = EXPECTED_LOG_PRIOR
            expected_post = expected_like + expected_prior
            if add_to_likelihood:
                expected_post += original_posteriors
            assert p1.size == 100
            assert np.allclose(p1, data[:, 0])
            assert np.allclose(p2, data[:, 1])
            assert np.allclose(output['PARAMETERS--P3'], p3)
            assert np.allclose(output["post"], expected_post)

        

@pytest.mark.skipif(sys.version_info < (3,7), reason="pocomc requires python3.6+")
def test_poco():
    try:
        import pocomc
    except ImportError:
        pytest.skip("pocomc not installed")
    run('pocomc', True, check_extra=False, n_effective=32, n_active=16,  n_total=32, n_evidence=32)

if __name__ == '__main__':
    import sys
    locals()[sys.argv[1]]()