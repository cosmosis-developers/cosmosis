from cosmosis.runtime import parameter
from cosmosis import run_cosmosis, Inifile
from cosmosis.output import InMemoryOutput
import numpy as np
import tempfile
import os

def test_override_new():
    with tempfile.NamedTemporaryFile('w') as values:
        values.write(
            "[parameters]\n"
            "p1=-3.0  0.0  3.0\n"
            "p2=-3.0  0.0  3.0\n"
            "p3=20.0\n")
        values.flush()
        print(values.name)
        params = parameter.Parameter.load_parameters(values.name, override=None)

    assert params[0] == ("parameters", "p1")
    assert params[0].start == 0.0
    assert np.allclose(params[0].limits, [-3, 3])

    assert params[1] == ("parameters", "p2")
    assert params[1].start == 0.0

    assert params[2] == ("parameters", "p3")
    assert np.isclose(params[2].start, 20.0)


    with tempfile.NamedTemporaryFile('w') as values:
        values.write(
            "[parameters]\n"
            "p1=-3.0  0.0  3.0\n"
            "p2=-3.0  0.0  3.0\n"
            "p3=20.0\n")
        values.flush()
        override = {
            ("parameters", "p1"): "1.0",
            ("xxx", "aaa"): "100.0",
        }
        params = parameter.Parameter.load_parameters(values.name, override=override)
    

    assert params[0] == ("parameters", "p1")
    assert params[0].start == 1.0
    assert np.allclose(params[0].limits, 1.0)

    assert params[1] == ("parameters", "p2")
    assert params[1].start == 0.0
    assert np.allclose(params[1].limits, [-3, 3])

    assert params[2] == ("parameters", "p3")
    assert np.isclose(params[2].start, 20.0)

    print(params)
    assert params[3] == ("xxx", "aaa")
    assert np.isclose(params[3].start, 100.0)
    assert np.allclose(params[3].limits, 100.0)


def test_header_update():
    root = os.path.split(os.path.abspath(__file__))[0]

    # Create a values file
    with tempfile.NamedTemporaryFile('w') as values:
        values.write(
            "[parameters]\n"
            "p1=-3.0  0.0  3.0\n"
            "p2=-3.0  0.0  3.0\n"
            "p3=20.0\n")
        values.flush()

        override = {
            ('runtime', 'root'): root,
            ('runtime', 'sampler'): 'emcee',
            ("pipeline", "debug"): "F",
            ("pipeline", "modules"): "test1",
            ("pipeline", "values"): values.name,
            ("test1", "file"): "example_module.py",
            ("emcee", "walkers"): "8",
            ("emcee", "samples"): "10"
        }
        ini = Inifile(None, override=override)
        output = InMemoryOutput()
        override_values = {
            ("parameters", "p1"): "-2.0  0.0  2.0",
        }

        run_cosmosis(ini, variables=override_values, output=output)
        print(output.comments)

        assert 'p1 = -2.0  0.0  2.0' in output.comments
