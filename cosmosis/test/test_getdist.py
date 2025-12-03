
import tempfile
import os
import cosmosis.getdist


def default_config():
    root = os.path.split(os.path.abspath(__file__))[0]

    params = {
        ('runtime', 'root'): root,
        ('runtime', 'verbosity'): "noisy",
        ("pipeline", "debug"): "F",
        ("pipeline", "modules"): "test1",
        ("pipeline", "extra_output"): "parameters/p3",
        ("test1", "file"): "example_module.py",
    }

    values = {
        ("parameters", "p1"): "-3.0 0.0 3.0",
        ("parameters", "p2"): "-3.0 0.0 3.0",
    }

    params = cosmosis.Inifile(None, override=params)
    values = cosmosis.Inifile(None, override=values)


    return params, values

def test_getdist_metropolis():
    with tempfile.TemporaryDirectory() as tempdir:
        params, values = default_config()
        params["runtime", "sampler"] = "metropolis"
        params.add_section("metropolis")
        params["metropolis", "samples"] = "1000"
        params["metropolis", "nsteps"] = "100"
        params.add_section("output")
        for i in [1, 2]:
            params["output", "filename"] = os.path.join(tempdir, f"chain_{i}.txt")
            cosmosis.run_cosmosis(params, values=values)

        chain_root = os.path.join(tempdir, "chain")
        cosmosis.getdist.cosmosis_to_getdist(chain_root, "metropolis", nchain=2, burn=0.1)

def test_getdist_emcee():
    with tempfile.TemporaryDirectory() as tempdir:
        params, values = default_config()
        params["runtime", "sampler"] = "emcee"
        params.add_section("emcee")
        params["emcee", "walkers"] = "16"
        params["emcee", "samples"] = "20"
        params["emcee", "nsteps"] = "5"
        params.add_section("output")
        params["output", "filename"] = os.path.join(tempdir, "chain.txt")
        cosmosis.run_cosmosis(params, values=values)

        chain_file = os.path.join(tempdir, "chain.txt")
        cosmosis.getdist.cosmosis_to_getdist(chain_file, "emcee", burn=0.1)

def test_getdist_nautilus():
    with tempfile.TemporaryDirectory() as tempdir:
        params, values = default_config()
        params["runtime", "sampler"] = "nautilus"
        params.add_section("nautilus")
        params["nautilus", "n_live"] = "100"
        params.add_section("output")
        params["output", "filename"] = os.path.join(tempdir, "chain.txt")
        cosmosis.run_cosmosis(params, values=values)

        chain_file = os.path.join(tempdir, "chain.txt")
        cosmosis.getdist.cosmosis_to_getdist(chain_file, "nautilus")
