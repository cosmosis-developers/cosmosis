from cosmosis.runtime import Inifile, register_new_parameter, LikelihoodPipeline, Parameter, Module
from cosmosis.datablock import DataBlock
from cosmosis.samplers.sampler import Sampler
from cosmosis.runtime.prior import TruncatedGaussianPrior, DeltaFunctionPrior
from cosmosis.output.in_memory_output import InMemoryOutput
from cosmosis.main import run_cosmosis, parser, mpi_pool
import numpy as np
import os
import tempfile
import pstats
import pytest

root = os.path.split(os.path.abspath(__file__))[0]

def test_add_param():

    sampler_class = Sampler.registry["emcee"]

    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-3.0  0.0  3.0\n"
        "p2=-3.0  0.0  3.0\n")
    values.flush()

    override = {
        ('runtime', 'root'): root,
        ("pipeline", "debug"): "F",
        ("pipeline", "modules"): "test2",
        ("pipeline", "values"): values.name,
        ("test2", "file"): "example_module2.py",
        ("emcee", "walkers"): "8",
        ("emcee", "samples"): "10"
    }

    ini = Inifile(None, override=override)

    # Make the pipeline itself
    pipeline = LikelihoodPipeline(ini)

    # test the the new added parameter has worked
    assert len(pipeline.varied_params) == 3
    p = pipeline.varied_params[2]
    assert str(p) == "new_parameters--p3"
    assert isinstance(p, Parameter)
    assert isinstance(p.prior, TruncatedGaussianPrior)
    assert np.isclose(p.prior.mu, 0.1)
    assert np.isclose(p.prior.sigma, 0.2)

    assert len(pipeline.fixed_params) == 1
    p = pipeline.fixed_params[0]
    assert isinstance(p, Parameter)
    assert isinstance(p.prior, DeltaFunctionPrior)


    output = InMemoryOutput()
    sampler = sampler_class(ini, pipeline, output)
    sampler.config()


    # check that the output is working
    assert output.column_index_for_name("new_parameters--p3") == 2

    while not sampler.is_converged():
        sampler.execute()


    p1 = output['parameters--p1']
    p2 = output['parameters--p2']
    p3 = output['new_parameters--p3']
    assert p3.max() < 1.0
    assert p3.min() > -1.0

def test_missing_setup():
    # check the register_new_parameter feature when no
    # setup is currently happening
    module = Module("test2", root + "/example_module2.py")
    config = DataBlock()
    module.setup(config)

def test_unused_param_warning(capsys):
    # check that an appropriate warning is generated
    # when a parameter is unused
    module = Module("test", root + "/example_module.py")
    config = DataBlock()
    config['test', 'unused'] = "unused_parameter"
    module.setup(config)
    out, _ = capsys.readouterr()
    assert "**** WARNING: Parameter 'unused'" in out

def test_vector_extra_outputs():
    with tempfile.TemporaryDirectory() as dirname:
        values_file = f"{dirname}/values.ini"
        params_file = f"{dirname}/params.ini"
        output_file = f"{dirname}/output.txt"
        with open(values_file, "w") as values:
            values.write(
                "[parameters]\n"
                "p1=-3.0  0.0  3.0\n"
                "p2=-3.0  0.0  3.0\n")

        params = {
            ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
            ('runtime', 'sampler'):  "emcee",
            ("pipeline", "debug"): "T",
            ("pipeline", "modules"): "test1",
            ("pipeline", "extra_output"): "data_vector/test_theory#2",
            ("pipeline", "values"): values_file,
            ("test1", "file"): "example_module.py",
            ("output", "filename"): output_file,
            ("emcee", "walkers"): "8",
            ("emcee", "samples"): "10",
        }

        ini = Inifile(None, override=params)
        status = run_cosmosis(ini)

        with open(output_file) as f:
            header = f.readline()

        assert "data_vector--test_theory_0" in header.lower()
        assert "data_vector--test_theory_1" in header.lower()

        data = np.loadtxt(output_file)
        # two parameters, two extra saves, prior, and posterior
        assert data.shape[1] == 6


def test_profile(capsys):
    with tempfile.TemporaryDirectory() as dirname:
        values_file = f"{dirname}/values.ini"
        params_file = f"{dirname}/params.ini"
        output_file = f"{dirname}/output.txt"
        stats_file  = f'{dirname}/profile.dat'
        with open(values_file, "w") as values:
            values.write(
                "[parameters]\n"
                "p1=-3.0  0.0  3.0\n"
                "p2=-3.0  0.0  3.0\n")

        params = {
            ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
            ('runtime', 'sampler'):  "emcee",
            ("pipeline", "debug"): "T",
            ("pipeline", "modules"): "test1",
            ("pipeline", "values"): values_file,
            ("test1", "file"): "example_module.py",
            ("output", "filename"): output_file,
            ("emcee", "walkers"): "8",
            ("emcee", "samples"): "10",
        }

        ini = Inifile(None, override=params)
        status = run_cosmosis(ini, profile_cpu=stats_file)

        output = capsys.readouterr()
        assert "cumtime" in output.out

        stats = pstats.Stats(stats_file)
        stats.sort_stats("cumtime")
        stats.print_stats(10)



def test_script_skip():
    with tempfile.TemporaryDirectory() as dirname:
        values_file = f"{dirname}/values.ini"
        params_file = f"{dirname}/params.ini"
        output_file = f"{dirname}/output.txt"
        with open(values_file, "w") as values:
            values.write(
                "[parameters]\n"
                "p1=-3.0  0.0  3.0\n"
                "p2=-3.0  0.0  3.0\n")

        params = {
            ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
            ('runtime', 'pre_script'): "this_executable_does_not_exist",
            ('runtime', 'post_script'): "this_executable_does_not_exist",
            ('runtime', 'sampler'):  "test",
            ("pipeline", "debug"): "F",
            ("pipeline", "modules"): "test1",
            ("pipeline", "values"): values_file,
            ("test1", "file"): "example_module.py",
            ("output", "filename"): output_file,
        }

        ini = Inifile(None, override=params)

        with pytest.raises(RuntimeError):
            status = run_cosmosis(ini)

        # shopuld work this time
        try:
            os.environ["COSMOSIS_NO_SUBPROCESS"] = "1"
            status = run_cosmosis(ini)
        finally:
            del os.environ["COSMOSIS_NO_SUBPROCESS"]

def test_prior_override():
    with tempfile.TemporaryDirectory() as dirname:
        values_file = f"{dirname}/values.ini"
        output_file = f"{dirname}/output.txt"
        with open(values_file, "w") as values:
            values.write(
                "[parameters]\n"
                "p1=-3.0  0.0  3.0\n"
                "p2=-3.0  0.0  3.0\n")

        params = {
            ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
            ('runtime', 'sampler'):  "apriori",
            ('apriori', 'nsample'):  "1000",
            ("pipeline", "debug"): "F",
            ("pipeline", "modules"): "test1",
            ("pipeline", "values"): values_file,
            ("test1", "file"): "example_module.py",
            ("output", "filename"): output_file,
        }

        args = parser.parse_args(["not_a_real_file"])
        ini = Inifile(None, override=params)

        priors_vals = {
            ('parameters', 'p1'): 'uniform -1.0    1.0'
        }

        priors = Inifile(None, override=priors_vals)

        status = run_cosmosis(ini, priors=priors)
        chain = np.loadtxt(output_file).T
        p1 = chain[0]
        p2 = chain[1]
        print(p1.min(), p1.max())
        assert p1.max() < 1.0
        assert p1.min() > -1.0

def test_pipeline_from_function():
    priors = {
        "p0": "gaussian 1.0 0.5",
    }

    def log_like(p):
        r1 = np.sum(np.abs(p))
        return -0.5 * np.sum(p**2), {"r1": r1}

    param_ranges = [
        (-3.0,  0.0,  3.0),
        (-3.0,  0.0,  3.0),
        (-3.0,  0.0,  3.0),
        (-3.0,  0.0,  3.0),
    ]

    derived = ["r1"]

    pipeline = LikelihoodPipeline.from_likelihood_function(log_like, param_ranges, priors=priors, derived=derived, debug=True)
    r = pipeline.run_results([0,0,0,0])
    assert r.like == 0.0
    assert r.extra[0] == 0.0

    # version without priors or derived params extracted
    pipeline = LikelihoodPipeline.from_likelihood_function(log_like, param_ranges)
    r = pipeline.run_results([0,0,0,0])
    assert r.like == 0.0
    assert len(r.extra) == 0


def test_failure_log():
    def log_like(p):
        if p[0] < 0:
            print("Deliberately failing pipeline ...")
            raise ValueError("p0 must be positive")
        return -0.5 * np.sum(p**2)

    param_ranges = [
        (-3.0,  0.0,  3.0),
        (-3.0,  0.0,  3.0),
        (-3.0,  0.0,  3.0),
        (-3.0,  0.0,  3.0),
    ]

    with tempfile.TemporaryDirectory() as dirname:
        logname = f"{dirname}/failure.log"
        failure_log = mpi_pool.MPILogFile(logname)
    
        pipeline = LikelihoodPipeline.from_likelihood_function(log_like, param_ranges)
        pipeline.set_failure_log_file(failure_log)
        # two that should work and produce nothing
        # and two that should log some errors
        param_sets = [
            [0.0,0.0,0.0,0.0],
            [1.0,0.0,0.0,0.0],
            [-1.0,0.0,0.0,4.0],
            [-2.0,0.0,0.0,8.0],
        ]
        print("Pipeline errors are expected below - we are testing logging of them")
        for p in param_sets:
            pipeline.run_parameters(p)

        failure_log.close()
        with open(logname) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].strip() == "-1.0 0.0 0.0 4.0"
        assert lines[1].strip() == "-2.0 0.0 0.0 8.0"


def test_recreate_pipeline():
    with tempfile.TemporaryDirectory() as dirname:

        values = os.path.join(dirname, "values.ini")
        priors = os.path.join(dirname, "priors.ini")
        output = os.path.join(dirname, "output.txt")
        with open(values, "w") as f:
            f.write(
                "[parameters]\n"
                "p1=-10.0  0.0  10.0\n"
                "p2=-1000.0  0.0  1000.0\n")

        with open(priors, "w") as f:
            f.write(
                "[parameters]\n"
                "p1=uniform -5.0 5.0\n"
                "p2=gaussian 0.0 1.0"
            )

        override = {
            ('runtime', 'root'): root,
            ('runtime', 'sampler'): "emcee",
            ("pipeline", "debug"): "F",
            ("pipeline", "modules"): "test2",
            ("pipeline", "values"): values,
            ("pipeline", "priors"): priors,
            ("pipeline", "extra_output"): "parameters/p3",
            ("output", "filename"): output,
            ("output", "format"): "text",
            ("test2", "file"): "example_module.py",
            ("emcee", "walkers"): "8",
            ("emcee", "samples"): "10"
        }

        ini = Inifile(None, override=override)

        status = run_cosmosis(ini)

        # Now we want to recreate the pipeline from the output
        pipeline = LikelihoodPipeline.from_chain_file(output)

        # check the basic pipeline configuration
        assert pipeline.modules[0].name == "test2"
        assert len(pipeline.modules) == 1

        # check it produces the same results
        r = pipeline.run_results([1.,2.])
        assert np.isclose(r.like, -2.5)
        assert np.isclose(r.extra[0], 3.0)

        # check that the priors have been correctly passed through
        # the recreated pipeline
        assert np.isclose(r.prior, np.log(0.1) - 2.0 - 0.5*np.log(2*np.pi))




if __name__ == '__main__':
    test_script_skip()
