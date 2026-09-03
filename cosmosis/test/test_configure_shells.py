import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_TREE_CONFIGURE_SCRIPT = ROOT_DIR / "bin" / "cosmosis-configure"


def csh_quote(value):
    return "'" + str(value).replace("'", "'\\''") + "'"


def run_shell(shell, command, env):
    shell_path = shutil.which(shell)
    if shell_path is None:
        pytest.skip(f"{shell} is not installed")

    args = [shell_path]
    if shell in {"csh", "tcsh"}:
        args.append("-f")
    args += ["-c", command]

    return subprocess.run(
        args,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def add_test_python_to_path(env, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python_link = bindir / "python"
    python_link.symlink_to(sys.executable)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"


def configure_script_path():
    if (ROOT_DIR / "setup.py").is_file() and SOURCE_TREE_CONFIGURE_SCRIPT.is_file():
        return SOURCE_TREE_CONFIGURE_SCRIPT

    script = shutil.which("cosmosis-configure")
    if script is not None:
        return Path(script)

    pytest.fail("Could not find bin/cosmosis-configure or installed cosmosis-configure")


@pytest.mark.parametrize("shell", ["bash", "zsh", "csh", "tcsh"])
def test_cosmosis_configure_sets_environment_in_shells(shell, tmp_path):
    source_dir = tmp_path / "cosmosis source"
    source_dir.mkdir()
    datablock_dir = source_dir / "datablock"

    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env["LIBRARY_PATH"] = "/existing/library"
    env["LD_LIBRARY_PATH"] = "/existing/ld-library"
    add_test_python_to_path(env, tmp_path)
    configure_script = configure_script_path()

    keys = [
        "COSMOSIS_SRC_DIR",
        "COSMOSIS_ALT_COMPILERS",
        "COSMOSIS_OMP",
        "COSMOSIS_DEBUG",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ]
    dump_environment = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote(
            "import json, os; "
            f"print(json.dumps({{key: os.environ.get(key) for key in {keys!r}}}))"
        )
    )

    if shell in {"csh", "tcsh"}:
        command = (
            f"source {csh_quote(configure_script)} "
            f"--source {csh_quote(source_dir)} --no-conda --debug; "
            "if ( $status != 0 ) exit $status; "
            f"{dump_environment}"
        )
    else:
        command = (
            f"source {shlex.quote(str(configure_script))} "
            f"--source {shlex.quote(str(source_dir))} --no-conda --debug; "
            "configure_status=$?; "
            "if [ $configure_status -ne 0 ]; then exit $configure_status; fi; "
            f"{dump_environment}"
        )

    result = run_shell(shell, command, env)
    configured_env = json.loads(result.stdout.strip().splitlines()[-1])

    assert configured_env["COSMOSIS_SRC_DIR"] == str(source_dir)
    assert configured_env["COSMOSIS_ALT_COMPILERS"] == "1"
    assert configured_env["COSMOSIS_OMP"] == "1"
    assert configured_env["COSMOSIS_DEBUG"] == "1"
    assert configured_env["LIBRARY_PATH"] == f"/existing/library:{datablock_dir}"
    assert configured_env["LD_LIBRARY_PATH"] == f"/existing/ld-library:{datablock_dir}"
