import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description="print commands that set up the cosmosis env")
cosmosis_src_dir = os.path.split(__file__)[0]
parser.add_argument("--source", default=cosmosis_src_dir)
parser.add_argument("--no-omp", action='store_false', dest='omp', help='Switch off OpenMP')
parser.add_argument("--omp", action='store_true', dest='omp', help='Switch on OpenMP (default)')
parser.add_argument("--debug", action='store_true', dest='debug', help='Switch on debug mode')
parser.add_argument("--no-debug", action='store_false', dest='debug', help='Switch on debug mode')
parser.add_argument("--no-conda", action='store_false', dest='conda', help='Switch off conda flags, even if conda env is found')
parser.add_argument("--brew", action='store_true', help='Print commands for homebrew with clang')
parser.add_argument("--brew-gcc", action='store_true', help='Print commands for homebrew with gcc')
parser.add_argument("--ports", action='store_true', help='Print commands for macports')
parser.add_argument("--automate-conda-setup", action='store_true', help='Automatically set up cosmosis when activating the environment from now on')

def homebrew_gfortran_libs():
    s = subprocess.run('gfortran -print-search-dirs', shell=True, capture_output=True)
    if s.returncode:
        return ""
    lines = s.stdout.decode().split("\n")

    for line in lines:
        if line.startswith("libraries:"):
            break
    else:
        return ""
    try:
        libdir = line.split("=")[1].split(":")[-1]
    except:
        return ""

    return f"-L {libdir}"

def homebrew_gcc_commands():
    s = subprocess.run('brew list --versions gcc', shell=True, capture_output=True)
    version = s.stdout.decode().split()[1].split('.')[0]
    return [
        f"export CC=gcc-{version}",
        f"export CXX=g++-{version}",
        f"export FC=gfortran-{version}",
        "export MPIFC=mpif90",
        "export COSMOSIS_ALT_COMPILERS=1",
    ]



def generate_commands(cosmosis_src_dir, debug=False, omp=True, brew=False, brew_gcc=False, conda=True, ports=False):
    conda = conda and ("CONDA_PREFIX" in os.environ)

    commands = [
        f"export COSMOSIS_SRC_DIR=\"{cosmosis_src_dir}\"",
        "export COSMOSIS_ALT_COMPILERS=1",
    ]

    if not brew:
        commands += [
            "export LIBRARY_PATH=$LIBRARY_PATH:$COSMOSIS_SRC_DIR/datablock",
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COSMOSIS_SRC_DIR/datablock",
    ]

    if brew:
        commands += [
            "export GSL_LIB=/usr/local/lib",
            "export GSL_INC=/usr/local/include",
            "export FFTW_LIBRARY=/usr/local/lib",
            "export FFTW_INCLUDE_DIR=/usr/local/include",
            "export LAPACK_LINK='-L /usr/local/opt/openblas/lib/ -l lapack'",
            "export LAPACK_LIB=/usr/local/opt/openblas/lib/",
            "export CFITSIO_LIB=/usr/local/lib",
            "export CFITSIO_INC=/usr/local/include",
        ]

        if brew_gcc:
            commands += homebrew_gcc_commands()
        else:
            commands += [
                f"export CC=clang",
                f"export CXX=clang++",
                f"export FC=gfortran",
                "export MPIFC=mpif90",
                "export COSMOSIS_ALT_COMPILERS=1",
            ]

    elif ports:
        commands += [
            'export GSL_INC=/opt/local/include',
            'export GSL_LIB=/opt/local/lib',
            'export CFITSIO_LIB=/opt/local/lib',
            'export CFITSIO_INC=/opt/local/include',
            'export FFTW_LIBRARY=/opt/local/lib',
            'export FFTW_INCLUDE_DIR=/opt/local/include',
            'export LAPACK_LINK="-L/opt/local/lib -llapack -lblas"',
            'export LAPACK_LIB=/opt/local/lib',
            'export CXX=/opt/local/bin/g++',
            'export CC=/opt/local/bin/gcc',
            'export FC=/opt/local/bin/gfortran',
            'export MPICC=mpicc',
            'export MPICXX=mpicxx',
            'export MPIFC=mpifort',
        ]

    elif conda:
        commands += [
            'export GSL_LIB=$CONDA_PREFIX/lib',
            'export GSL_INC=$CONDA_PREFIX/include',
            'export FFTW_LIBRARY=$CONDA_PREFIX/lib',
            'export FFTW_INCLUDE_DIR=$CONDA_PREFIX/include',
            'export LAPACK_LINK="-L$CONDA_PREFIX/lib -llapack"',
            'export LAPACK_LIB="$CONDA_PREFIX/lib"',
            'export CFITSIO_LIB=$CONDA_PREFIX/lib',
            'export CFITSIO_INC=$CONDA_PREFIX/include',
            ]

    if omp:
        commands.append("export COSMOSIS_OMP=1")
        
    if debug:
        commands.append("COSMOSIS_DEBUG=1")

    return commands

def automate_conda_setup(cosmosis_src_dir, cmds, conda):
    if not (conda and "CONDA_PREFIX" in os.environ):
        print("Error: --automate-conda-setup requires a conda environment to be active", file=sys.stderr)
        sys.exit(1)

    activate_dir = os.path.join(os.environ["CONDA_PREFIX"], "etc", "conda", "activate.d")
    os.makedirs(activate_dir, exist_ok=True)
    activate_script_path =  os.path.join(activate_dir, "activate_cosmosis.sh")

    with open(activate_script_path, 'w') as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    deactivate_dir = os.path.join(os.environ["CONDA_PREFIX"], "etc", "conda", "deactivate.d")
    os.makedirs(deactivate_dir, exist_ok=True)
    deactivate_script_path = os.path.join(deactivate_dir, "deactivate_cosmosis.sh")

    with open(deactivate_script_path, 'w') as f:
        f.write("unset COSMOSIS_SRC_DIR\n")
        f.write("unset COSMOSIS_ALT_COMPILERS\n")
        f.write("unset GSL_LIB\n")
        f.write("unset GSL_INC\n")
        f.write("unset FFTW_LIBRARY\n")
        f.write("unset FFTW_INCLUDE_DIR\n")
        f.write("unset LAPACK_LINK\n")
        f.write("unset LAPACK_LIB\n")
        f.write("unset CFITSIO_LIB\n")
        f.write("unset CFITSIO_INC\n")
        f.write("unset COSMOSIS_OMP\n")
        f.write("unset COSMOSIS_DEBUG\n")

        v = f"{cosmosis_src_dir}/datablock"
        # remove datablock from LIBRARY_PATH and LD_LIBRARY_PATH
        f.write('export LIBRARY_PATH=$(echo $LIBRARY_PATH | tr ":" "\\n" | grep -v "{}" | tr "\\n" ":")\n'.format(v))
        f.write('export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\\n" | grep -v "{}" | tr "\\n" ":")\n'.format(v))

    print(f"CosmoSIS will automatically configure when you activate the environment from now on", file=sys.stderr)


if __name__ == '__main__':
    args = parser.parse_args()
    cmds = generate_commands(args.source, debug=args.debug, omp=args.omp, conda=args.conda, brew=args.brew or args.brew_gcc, brew_gcc=args.brew_gcc, ports=args.ports)
    if args.automate_conda_setup:
        automate_conda_setup(args.source, cmds, args.conda)
    print("; ".join(cmds))

