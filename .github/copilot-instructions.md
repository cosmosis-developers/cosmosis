# CosmoSIS
CosmoSIS is a cosmological parameter estimation framework written in Python with C/C++/Fortran components. It provides a modular architecture for running cosmological analyses and includes various sampling algorithms for Bayesian parameter estimation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository:
- Install system dependencies:
  - `sudo apt-get update && sudo apt-get -y install gfortran-11 swig libopenmpi-dev openmpi-bin libopenblas-dev` -- takes 2-3 minutes. NEVER CANCEL. Set timeout to 300+ seconds.
  - `sudo ln -sf $(which gfortran-11) /usr/local/bin/gfortran` (if needed)
- Install Python dependencies:
  - `python -m pip install --upgrade pip`
  - `pip install -vv .` -- takes 75 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- Configure environment (REQUIRED before building):
  - `source bin/cosmosis-configure` -- sets COSMOSIS_SRC_DIR and other environment variables
- Build native libraries:
  - `cd cosmosis && make` -- takes 40 seconds from clean. NEVER CANCEL. Set timeout to 120+ seconds.
- Run tests:
  - Install test dependencies: `pip install astropy pytest-cov getdist derivative numdifftools "pocomc==1.2.6"`
  - `export COSMOSIS_TESTING=1 && mkdir -p tmp && cd tmp && pytest --pyargs cosmosis` -- takes 78 seconds. NEVER CANCEL. Set timeout to 180+ seconds.

### Build from Clean:
- Clean build artifacts: `cd cosmosis && make clean && python setup.py clean`
- Rebuild: `cd cosmosis && make` -- takes 40 seconds from clean

## Validation

- ALWAYS manually validate that the application works after making changes:
  - `cosmosis --version` -- should show version number (e.g., 3.21.1)
  - `cosmosis cosmosis/test/example.ini` -- should run successfully and show Prior/Likelihood/Posterior values
  - `cosmosis-campaign --help` -- should show campaign management help
  - `cosmosis-postprocess --help` -- should show postprocessing help
- ALWAYS run through at least one complete end-to-end scenario after making changes.
- Run the test suite to ensure nothing is broken.
- Test with different samplers if modifying sampler code.

## Common Tasks

### Repository Structure
```
cosmosis/                 # Main Python package directory
├── __init__.py          # Package initialization
├── main.py              # Main entry point
├── campaign.py          # Campaign management
├── configure.py         # Configuration system
├── datablock/           # C/C++/Fortran datablock library
├── samplers/            # Sampling algorithms (emcee, multinest, polychord, etc.)
├── runtime/             # Runtime system and parameter handling
├── test/                # Test suite
├── Makefile             # Build system for native components
└── config/              # Build configuration

bin/                     # Command-line tools
├── cosmosis             # Main cosmosis command
├── cosmosis-configure   # Environment configuration script
├── cosmosis-campaign    # Campaign management tool
└── cosmosis-postprocess # Post-processing tool

examples/                # Example configurations
modules/                 # Empty (placeholder for standard library)
```

### Key Commands
- `cosmosis <inifile>` -- Run a pipeline with parameters from inifile
- `cosmosis-campaign --list <yamlfile>` -- List available runs in campaign
- `cosmosis-campaign --run <runname> <yamlfile>` -- Run specific campaign
- `cosmosis-postprocess <inifile>` -- Generate plots and statistics from output
- `source bin/cosmosis-configure` -- Set up build environment (REQUIRED)

### Running CosmoSIS
1. ALWAYS source the configuration script first: `source bin/cosmosis-configure`
2. Run with an ini file: `cosmosis path/to/config.ini`
3. Test with the example: `cosmosis cosmosis/test/example.ini`

### Testing and Development
- Run full test suite: `export COSMOSIS_TESTING=1 && cd tmp && pytest --pyargs cosmosis`
- Run specific tests: `pytest --pyargs cosmosis.test.test_samplers -k "test_emcee"`
- Skip problematic tests: Use `-k "not mpi"` or set environment variables like `SKIP_NAUTILUS=1`

### Build System
- The project uses both Python setuptools (`pip install`) and Makefiles for native components
- ALWAYS run `source bin/cosmosis-configure` before building with make
- Native libraries are built in `cosmosis/datablock/` and `cosmosis/samplers/`
- Clean with: `cd cosmosis && make clean`

### Configuration System
- `bin/cosmosis-configure` is a bash script that sets environment variables
- It calls `python -m cosmosis.configure` to generate the commands
- Key environment variables: `COSMOSIS_SRC_DIR`, `COSMOSIS_OMP`, `LIBRARY_PATH`, `LD_LIBRARY_PATH`
- Options: `--brew`, `--brew-gcc`, `--ports`, `--no-conda`, `--debug`, `--omp/--no-omp`

## Timing Expectations and Critical Warnings

- **NEVER CANCEL** any build or test commands. They may appear to hang but are working.
- **pip install**: 75 seconds normal, up to 2 minutes with complex dependencies
- **make build**: 40 seconds from clean, nearly instant for incremental builds  
- **pytest**: 78 seconds for full suite, varies by number of tests selected
- **System dependency install**: 2-3 minutes on fresh systems

Always set timeouts of 120+ seconds for builds and 180+ seconds for tests.

## Dependencies and Environment

### Required System Packages
- `gfortran-11` (or compatible Fortran compiler)
- `swig` (for interface generation)
- `libopenmpi-dev openmpi-bin` (MPI support)
- `libopenblas-dev` (linear algebra)

### Required Python Packages
Core: `numpy`, `scipy`, `matplotlib`, `pyyaml`, `emcee`, `pybind11`, `dynesty`, `zeus-mcmc`, `nautilus-sampler`, `scikit-learn`

Testing: `pytest`, `astropy`, `pytest-cov`, `getdist`, `derivative`, `numdifftools`

Optional: `mpi4py`, `pocomc==1.2.6`

### Environment Variables
After running `source bin/cosmosis-configure`:
- `COSMOSIS_SRC_DIR` -- Points to cosmosis package directory
- `COSMOSIS_OMP=1` -- Enables OpenMP support
- `COSMOSIS_ALT_COMPILERS=1` -- Uses alternative compiler settings
- `LIBRARY_PATH` and `LD_LIBRARY_PATH` -- Include datablock library

## Troubleshooting

### Common Issues
- "ModuleNotFoundError": Install missing Python dependencies
- "COSMOSIS_SRC_DIR not set": Run `source bin/cosmosis-configure`
- Build failures: Check that system dependencies are installed
- Test failures: Some tests require optional dependencies (skip with `-k "not testname"`)

### Known Limitations
- Minuit sampler requires additional Minuit2 library (optional)
- MPI functionality requires properly configured mpi4py (optional)
- Some tests may be skipped on systems without specific dependencies

## Samplers and Modules

CosmoSIS includes numerous sampling algorithms:
- `emcee` -- Ensemble MCMC sampler
- `multinest` -- MultiNest nested sampling
- `polychord` -- PolyChord nested sampling  
- `dynesty` -- Dynamic nested sampling
- `zeus` -- Zeus ensemble sampler
- `nautilus` -- Nautilus nested sampler
- `metropolis` -- Metropolis-Hastings MCMC
- `fisher` -- Fisher matrix estimation
- `grid` -- Grid sampling
- `test` -- Test sampler for validation

Configure samplers in pipeline ini files under `[runtime]` section with `sampler = <name>`.