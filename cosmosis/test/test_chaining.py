from cosmosis.runtime.config import Inifile
from cosmosis.main import run_cosmosis, parser
import tempfile
import os
import sys
import pytest
import numpy as np


def test_sampler_chain():


    with tempfile.TemporaryDirectory() as dirname:
        values_file = f"{dirname}/values.txt"

        # we run maxlike twice and check the files are different
        maxlike_file0 = f"{dirname}/chain.maxlike.0.txt"
        maxlike_file1 = f"{dirname}/chain.maxlike.1.txt"
        fisher_file = f"{dirname}/chain.fisher.txt"
        emcee_file = f"{dirname}/chain.txt"
        with open(values_file, "w") as values:
            values.write(
                "[parameters]\n"
                "p1=-3.0  0.0  3.0\n"
                "p2=-3.0  0.0  3.0\n")

        params = {
            ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
            ('runtime', 'sampler'):  "maxlike maxlike fisher emcee",
            ("pipeline", "debug"): "T",
            ("pipeline", "modules"): "test1",
            ("pipeline", "extra_output"): "parameters/p3",
            ("pipeline", "values"): values_file,
            ("test1", "file"): "example_module.py",
            ("output", "filename"): emcee_file,
            ("emcee", "walkers"): "8",
            ("emcee", "samples"): "100",
            ("maxlike", "tolerance"): "0.05",
            ("fisher", "step_size"): "0.01"
        }


        ini = Inifile(None, override=params)

        status = run_cosmosis(ini)

        data = np.loadtxt(fisher_file)
        print(data.shape)

        data = np.loadtxt(maxlike_file0)
        print(data.shape)
        data = np.loadtxt(maxlike_file1)
        print(data.shape)

        data = np.loadtxt(emcee_file)
        print(data.shape)



    assert status == 0

