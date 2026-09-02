import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGURE_SCRIPT = ROOT_DIR / "bin" / "cosmosis-configure"


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


@pytest.mark.parametrize("shell", ["bash", "zsh", "csh", "tcsh"])
def test_cosmosis_configure_sets_environment_in_shells(shell, tmp_path):
    source_dir = tmp_path / "cosmosis source"
    source_dir.mkdir()
    datablock_dir = source_dir / "datablock"

    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env["LIBRARY_PATH"] = "/existing/library"
    env["LD_LIBRARY_PATH"] = "/existing/ld-library"

    keys = [
        "COSMOSIS_SRC_DIR",
        "COSMOSIS_ALT_COMPILERS",
        "COSMOSIS_OMP",
        "COSMOSIS_DEBUG",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ]
    dump_environment = (
        "python -c "
        + shlex.quote(
            "import json, os; "
            f"print(json.dumps({{key: os.environ.get(key) for key in {keys!r}}}))"
        )
    )

    if shell in {"csh", "tcsh"}:
        command = (
            f"source {csh_quote(CONFIGURE_SCRIPT)} "
            f"--source {csh_quote(source_dir)} --no-conda --debug; "
            f"{dump_environment}"
        )
    else:
        command = (
            f"source {shlex.quote(str(CONFIGURE_SCRIPT))} "
            f"--source {shlex.quote(str(source_dir))} --no-conda --debug; "
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
