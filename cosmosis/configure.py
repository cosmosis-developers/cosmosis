import os
import sys
import argparse
import shlex
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
parser.add_argument("--shell", choices=["auto", "sh", "csh"], default="auto", help="Shell syntax to print")


CSH_SHELLS = {"csh", "tcsh"}
SH_SHELLS = {"sh", "bash", "zsh", "ksh"}

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

def homebrew_gcc_vars():
    s = subprocess.run('brew list --versions gcc', shell=True, capture_output=True)
    version = s.stdout.decode().split()[1].split('.')[0]
    return [
        ("CC", f"gcc-{version}"),
        ("CXX", f"g++-{version}"),
        ("FC", f"gfortran-{version}"),
        ("MPIFC", "mpif90"),
        ("COSMOSIS_ALT_COMPILERS", "1"),
    ]


def shell_name(command):
    return os.path.basename(command).lstrip("-")


def detect_shell():
    try:
        parent = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "comm="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        parent = ""

    name = shell_name(parent)
    if name in CSH_SHELLS:
        return "csh"
    if name in SH_SHELLS:
        return "sh"

    name = shell_name(os.environ.get("SHELL", ""))
    if name in CSH_SHELLS:
        return "csh"
    return "sh"


def csh_quote(value):
    return "'" + value.replace("'", "'\\''") + "'"


def render_commands(variables, shell):
    if shell == "auto":
        shell = detect_shell()

    if shell == "csh":
        return [f"setenv {name} {csh_quote(value)}" for name, value in variables]
    else:
        return [f"export {name}={shlex.quote(value)}" for name, value in variables]


def append_path(name, value):
    return os.environ.get(name, "") + ":" + value


def generate_commands(cosmosis_src_dir, debug=False, omp=True, brew=False, brew_gcc=False, conda=True, ports=False, shell="auto"):
    conda = conda and ("CONDA_PREFIX" in os.environ)

    variables = [
        ("COSMOSIS_SRC_DIR", cosmosis_src_dir),
        ("COSMOSIS_ALT_COMPILERS", "1"),
    ]

    if not brew:
        datablock_dir = os.path.join(cosmosis_src_dir, "datablock")
        variables += [
            ("LIBRARY_PATH", append_path("LIBRARY_PATH", datablock_dir)),
            ("LD_LIBRARY_PATH", append_path("LD_LIBRARY_PATH", datablock_dir)),
        ]

    if brew:
        variables += [
            ("GSL_LIB", "/usr/local/lib"),
            ("GSL_INC", "/usr/local/include"),
            ("FFTW_LIBRARY", "/usr/local/lib"),
            ("FFTW_INCLUDE_DIR", "/usr/local/include"),
            ("LAPACK_LINK", "-L /usr/local/opt/openblas/lib/ -l lapack"),
            ("LAPACK_LIB", "/usr/local/opt/openblas/lib/"),
            ("CFITSIO_LIB", "/usr/local/lib"),
            ("CFITSIO_INC", "/usr/local/include"),
        ]

        if brew_gcc:
            variables += homebrew_gcc_vars()
        else:
            variables += [
                ("CC", "clang"),
                ("CXX", "clang++"),
                ("FC", "gfortran"),
                ("MPIFC", "mpif90"),
                ("COSMOSIS_ALT_COMPILERS", "1"),
            ]

    elif ports:
        variables += [
            ("GSL_INC", "/opt/local/include"),
            ("GSL_LIB", "/opt/local/lib"),
            ("CFITSIO_LIB", "/opt/local/lib"),
            ("CFITSIO_INC", "/opt/local/include"),
            ("FFTW_LIBRARY", "/opt/local/lib"),
            ("FFTW_INCLUDE_DIR", "/opt/local/include"),
            ("LAPACK_LINK", "-L/opt/local/lib -llapack -lblas"),
            ("LAPACK_LIB", "/opt/local/lib"),
            ("CXX", "/opt/local/bin/g++"),
            ("CC", "/opt/local/bin/gcc"),
            ("FC", "/opt/local/bin/gfortran"),
            ("MPICC", "mpicc"),
            ("MPICXX", "mpicxx"),
            ("MPIFC", "mpifort"),
        ]

    elif conda:
        conda_prefix = os.environ["CONDA_PREFIX"]
        variables += [
            ("GSL_LIB", os.path.join(conda_prefix, "lib")),
            ("GSL_INC", os.path.join(conda_prefix, "include")),
            ("FFTW_LIBRARY", os.path.join(conda_prefix, "lib")),
            ("FFTW_INCLUDE_DIR", os.path.join(conda_prefix, "include")),
            ("LAPACK_LINK", f"-L{os.path.join(conda_prefix, 'lib')} -llapack"),
            ("LAPACK_LIB", os.path.join(conda_prefix, "lib")),
            ("CFITSIO_LIB", os.path.join(conda_prefix, "lib")),
            ("CFITSIO_INC", os.path.join(conda_prefix, "include")),
            ]

    if omp:
        variables.append(("COSMOSIS_OMP", "1"))
        
    if debug:
        variables.append(("COSMOSIS_DEBUG", "1"))

    return render_commands(variables, shell)

def automate_conda_setup(cosmosis_src_dir, cmds, conda, shell):
    if not (conda and "CONDA_PREFIX" in os.environ):
        print("Error: --automate-conda-setup requires a conda environment to be active", file=sys.stderr)
        sys.exit(1)

    activate_dir = os.path.join(os.environ["CONDA_PREFIX"], "etc", "conda", "activate.d")
    os.makedirs(activate_dir, exist_ok=True)
    extension = "csh" if shell == "csh" else "sh"
    activate_script_path =  os.path.join(activate_dir, f"activate_cosmosis.{extension}")

    with open(activate_script_path, 'w') as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    deactivate_dir = os.path.join(os.environ["CONDA_PREFIX"], "etc", "conda", "deactivate.d")
    os.makedirs(deactivate_dir, exist_ok=True)
    deactivate_script_path = os.path.join(deactivate_dir, f"deactivate_cosmosis.{extension}")

    variables = [
        "COSMOSIS_SRC_DIR",
        "COSMOSIS_ALT_COMPILERS",
        "GSL_LIB",
        "GSL_INC",
        "FFTW_LIBRARY",
        "FFTW_INCLUDE_DIR",
        "LAPACK_LINK",
        "LAPACK_LIB",
        "CFITSIO_LIB",
        "CFITSIO_INC",
        "COSMOSIS_OMP",
        "COSMOSIS_DEBUG",
    ]

    datablock_dir = os.path.join(cosmosis_src_dir, "datablock")
    with open(deactivate_script_path, 'w') as f:
        if shell == "csh":
            for variable in variables:
                f.write(f"unsetenv {variable}\n")
            for path_variable in ["LIBRARY_PATH", "LD_LIBRARY_PATH"]:
                f.write(f"if ( $?{path_variable} ) then\n")
                f.write(
                    f"    set _cosmosis_path = `python -c 'import sys; print(\":\".join(p for p in sys.argv[1].split(\":\") if p != sys.argv[2]))' \"${path_variable}\" {csh_quote(datablock_dir)}`\n"
                )
                f.write(f"    setenv {path_variable} \"$_cosmosis_path\"\n")
                f.write("    unset _cosmosis_path\n")
                f.write("endif\n")
        else:
            for variable in variables:
                f.write(f"unset {variable}\n")
            # remove datablock from LIBRARY_PATH and LD_LIBRARY_PATH
            f.write('export LIBRARY_PATH=$(echo $LIBRARY_PATH | tr ":" "\\n" | grep -v "{}" | tr "\\n" ":")\n'.format(datablock_dir))
            f.write('export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\\n" | grep -v "{}" | tr "\\n" ":")\n'.format(datablock_dir))

    print(f"CosmoSIS will automatically configure when you activate the environment from now on", file=sys.stderr)


if __name__ == '__main__':
    args = parser.parse_args()
    cmds = generate_commands(args.source, debug=args.debug, omp=args.omp, conda=args.conda, brew=args.brew or args.brew_gcc, brew_gcc=args.brew_gcc, ports=args.ports, shell=args.shell)
    if args.automate_conda_setup:
        automate_conda_setup(args.source, cmds, args.conda, detect_shell() if args.shell == "auto" else args.shell)
    print("; ".join(cmds))
