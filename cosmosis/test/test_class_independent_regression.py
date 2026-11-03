import importlib.util
import os
from pathlib import Path

import numpy as np

from cosmosis.datablock import DataBlock, names


def load_class_interface():
    path = Path('/work/standard-library/boltzmann/class/class_interface.py')
    spec = importlib.util.spec_from_file_location('class_interface_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_class_outputs_match_classy():
    import classy

    interface = load_class_interface()
    solver = classy.Class()
    solver.set({
        'h': 0.7, 'omega_b': 0.022, 'omega_cdm': 0.12,
        'A_s': 2.1e-9, 'n_s': 0.965, 'output': 'mPk',
        'non linear': 'halofit', 'P_k_max_h/Mpc': 2.0, 'z_max_pk': 0.2,
        'N_ncdm': 1, 'm_ncdm': 0.06, 'N_ur': 2.0328,
    })
    class_root = '/work/standard-library/boltzmann/class/class_v3.2.5'
    previous_cwd = os.getcwd()
    os.chdir(class_root)
    try:
        solver.compute()
    finally:
        os.chdir(previous_cwd)
    config = {
        'kmax': 1.0, 'nk': 8, 'zmax': 0.1,
        'background_only': False, 'cmb': False,
        'save_matter_power_lin': True,
        'save_cdm_baryon_power_lin': True,
    }
    block = DataBlock()
    block[names.cosmological_parameters, 'h0'] = 0.7
    interface.get_class_outputs(block, solver, config)
    z, kh, nonlinear = block.get_grid('matter_power_nl', 'z', 'k_h', 'p_k')
    expected = np.array([
        [solver.pk(float(k * 0.7), float(zi)) * 0.7**3 for k in kh]
        for zi in z
    ])
    np.testing.assert_array_equal(nonlinear, expected)
    assert block.has_section('matter_power_lin')
    assert block.has_section('cdm_baryon_power_lin')
    assert np.isfinite(nonlinear).all()
    solver.struct_cleanup()
    solver.empty()
