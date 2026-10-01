from pathlib import Path
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(command, env=None):
    print("\n>", " ".join(str(x) for x in command))
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Build and verify pyMKM source and wheel distributions."
    )
    parser.add_argument(
        "--force-trusted",
        action="store_true",
        help=(
            "Trust pypi.org and files.pythonhosted.org for pip operations "
            "performed by isolated temporary environments."
        ),
    )
    args = parser.parse_args()

    build_env = os.environ.copy()

    if args.force_trusted:
        print(
            "WARNING: trusted-host mode enabled for "
            "pypi.org and files.pythonhosted.org."
        )
        build_env["PIP_TRUSTED_HOST"] = "pypi.org files.pythonhosted.org"

    if DIST.exists():
        shutil.rmtree(DIST)

    print("[1/3] Building source and wheel distributions...")
    run([sys.executable, "-m", "build"], env=build_env)

    artifacts = sorted(DIST.glob("*"))
    if not artifacts:
        raise RuntimeError("No distribution artifacts were generated.")

    print("\n[2/3] Verifying distributions with twine...")

    if importlib.util.find_spec("twine") is not None:
        run(
            [sys.executable, "-m", "twine", "check", *artifacts],
            env=build_env,
        )
    else:
        print("Twine is not installed in the current environment.")
        print("Creating a temporary isolated environment for verification...")

        with tempfile.TemporaryDirectory(prefix="pymkm-twine-") as tmpdir:
            venv_dir = Path(tmpdir) / "venv"
            run([sys.executable, "-m", "venv", str(venv_dir)])

            if os.name == "nt":
                temp_python = venv_dir / "Scripts" / "python.exe"
            else:
                temp_python = venv_dir / "bin" / "python"

            pip_command = [
                temp_python,
                "-m",
                "pip",
                "install",
            ]

            if args.force_trusted:
                pip_command.extend(
                    [
                        "--trusted-host",
                        "pypi.org",
                        "--trusted-host",
                        "files.pythonhosted.org",
                    ]
                )

            pip_command.append("twine")
            run(pip_command, env=build_env)

            run(
                [
                    temp_python,
                    "-m",
                    "twine",
                    "check",
                    *artifacts,
                ],
                env=build_env,
            )

    print("\n[3/3] Distribution contents:")
    for artifact in artifacts:
        print(f"\n--- {artifact.name} ---")

        if artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact, "r:gz") as archive:
                for name in archive.getnames():
                    print(name)

        elif artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                for name in archive.namelist():
                    print(name)

    print("\nBuild and verification completed successfully.")


if __name__ == "__main__":
    main()
