import setuptools
import setuptools.command.install
import setuptools.command.build_py
import setuptools.command.develop
import subprocess
import os

version = open('cosmosis/version.py').read().split('=')[1].strip().strip("'")


f90_mods = [
    "datablock/cosmosis_section_names.mod",
    "datablock/cosmosis_types.mod",
    "datablock/cosmosis_wrappers.mod",
    "datablock/cosmosis_modules.mod",
]

scripts = [
    'bin/cosmosis',
    'bin/cosmosis-campaign',
    'bin/cosmosis-configure',
    'bin/cosmosis-extract',
    'bin/cosmosis-sample-fisher',
    'bin/cosmosis-postprocess',
]

c_headers = [
    "datablock/c_datablock.h",
    "datablock/datablock_logging.h",
    "datablock/datablock_types.h",
    "datablock/cosmosis_constants.h",
    "datablock/datablock_status.h",
    "datablock/section_names.h",
]

cc_headers = [
    "datablock/clamp.hh",
    "datablock/entry.hh",
    "datablock/fakearray.hh",
    "datablock/ndarray.hh",
    "datablock/datablock.hh",
    "datablock/exceptions.hh",
    "datablock/mdarraygen.hh",
    "datablock/section.hh"
]

datablock_libs = ["datablock/libcosmosis.so"]



if os.environ.get("COSMOSIS_JULIA_FLAGS", ""):
    julia_files = ["runtime/julia_modules/libcosmosis_julia.so", "datablock/julia/cosmosis.jl"]
else:
    julia_files = []

sampler_libs = ["samplers/multinest/multinest_src/libnest3.so",
                "samplers/multinest/multinest_src/libnest3_mpi.so",
                "samplers/polychord/polychord_src/libchord.so",
                "samplers/polychord/polychord_src/libchord_mpi.so",
                "samplers/minuit/minuit_wrapper.so"]


testing_files = [
    "test/libtest/c_datablock_complex_array_test.c",
    "test/libtest/c_datablock_double_array_test.c",
    "test/libtest/c_datablock_int_array_test.c",
    "test/libtest/c_datablock_multidim_complex_array_test.c",
    "test/libtest/c_datablock_multidim_double_array_test.c",
    "test/libtest/c_datablock_multidim_int_array_test.c",
    "test/libtest/c_datablock_test.c",
    "test/libtest/cosmosis_test.F90",
    "test/libtest/cosmosis_tests.supp",
    "test/libtest/datablock_test.cc",
    "test/libtest/entry_test.cc",
    "test/libtest/ndarray_test.cc",
    "test/libtest/section_test.cc",
    "test/libtest/test_c_datablock_scalars.h",
    "test/libtest/test_c_datablock_scalars.template",
    "test/libtest/Makefile",
    "test/campaign.yml",
    "test/included.yml",
    "test/bad-campaign.yml",
    "test/example-priors.ini",
    "test/example-values.ini",
    "test/example.ini",

]

other_files = ["postprocessing/latex.ini"]

compilers_config = ["config/compilers.mk", "config/subdirs.mk"]


def get_COSMOSIS_SRC_DIR():
    cosmosis_src_dir = os.path.join(os.getcwd(), "cosmosis")
    return cosmosis_src_dir

def make_cosmosis():
    print("Running CosmoSIS main library compile command")
    cosmosis_src_dir = get_COSMOSIS_SRC_DIR()
    env = os.environ.copy()
    env["COSMOSIS_SRC_DIR"] = cosmosis_src_dir

    # If we are in a conda build env then the appropriate
    # variable is called PREFIX. Otherwise if we are on
    # a user's build env then it is called CONDA_PREFIX
    if os.environ.get("CONDA_BUILD", "0") == "1":
        prefix_name = "PREFIX"
        conda_build = True
    else:
        prefix_name = "CONDA_PREFIX"
        conda_build = False

    conda = env.get(prefix_name)
    if conda:
        # User can switch on COSMOSIS_OMP manually, but it should
        # always be on for conda.
        env["COSMOSIS_OMP"] = "1"

        # We also need Lapack to build some of the samplers
        env["LAPACK_LINK"] = f"-L{conda}/lib -llapack"
        # and minuit for that sampler
        env['MINUIT2_LIB'] = f"{conda}/lib"
        env['MINUIT2_INC'] = f"{conda}/include/Minuit2"

        if conda_build:
            env["COSMOSIS_ALT_COMPILERS"] = "1"
            env['USER_FFLAGS'] = env['FFLAGS']
            env['USER_CFLAGS'] = env['CFLAGS']
            env['USER_CXXFLAGS'] = env['CXXFLAGS']

        # and the MPI compiler
        env["MPIFC"] = "mpif90"

    env['FC'] = env.get('FC', 'gfortran')

    subprocess.check_call(["make"], env=env, cwd="cosmosis")

class build_cosmosis(setuptools.Command):
    description = "Build CosmoSIS and do nothing else"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        make_cosmosis()



class build_py_cosmosis(setuptools.command.build_py.build_py):
    def run(self):
        make_cosmosis()
        super().run()

class clean_cosmosis(setuptools.Command):
    description = "Run the CosmoSIS clean process"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("Running CosmoSIS main library clean command")
        cosmosis_src_dir = get_COSMOSIS_SRC_DIR()
        env = {"COSMOSIS_SRC_DIR": cosmosis_src_dir,}
        subprocess.check_call(["make", "clean"], env=env, cwd="cosmosis")

class install_cosmosis(setuptools.command.install.install):
    description = "Run the CosmoSIS install process"

    def run(self):
        make_cosmosis()
        super().run()

class develop_cosmosis(setuptools.command.develop.develop):
    description = "Install CosmoSIS in editable mode"
    def run(self):
        make_cosmosis()
        super().run()


requirements = [
    "pyyaml",
    "emcee",
    "numpy",
    "scipy",
    "matplotlib",
    "pybind11",
    "pyyaml",
    "scipy",
    "threadpoolctl",
    "emcee",
    "dynesty",
    "zeus-mcmc",
    "nautilus-sampler>=1.0.1",
    "h5py", # needed for nautilus
    "dulwich",
    "scikit-learn",
    "Py-BOBYQA",

]

all_package_files = (datablock_libs + sampler_libs
                            + c_headers + cc_headers + f90_mods 
                            + compilers_config + testing_files + other_files + julia_files)

setuptools.setup(name = 'cosmosis',
    description       = "The CosmoSIS parameter estimation library.",
    author            = "Joe Zuntz",
    author_email      = "joezuntz@googlemail.com",
    url               = "https://github.com/joezuntz/cosmosis",
    packages = setuptools.find_packages(),
    package_data = {"cosmosis" : all_package_files},
    scripts = scripts,
    install_requires = requirements,
    cmdclass={
        "build_cosmosis": build_cosmosis,
        "build_py": build_py_cosmosis,
        "install": install_cosmosis,
        "develop": develop_cosmosis,
        "clean": clean_cosmosis,
    },
    version=version,
)

