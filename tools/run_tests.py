from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("\n>", " ".join(str(x) for x in command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the pyMKM test suite, optionally installing dev dependencies first."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install pyMKM in editable mode with dev dependencies before running tests.",
    )
    parser.add_argument(
        "--force-trusted",
        action="store_true",
        help=(
            "Trust pypi.org and files.pythonhosted.org during installation. "
            "Requires --install."
        ),
    )
    args = parser.parse_args()

    if args.force_trusted and not args.install:
        parser.error("--force-trusted requires --install")

    if args.install:
        print("[1/2] Installing pyMKM in editable mode with dev dependencies...")

        pip_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
        ]

        if args.force_trusted:
            print(
                "WARNING: pip trusted-host mode enabled for "
                "pypi.org and files.pythonhosted.org."
            )
            pip_command.extend([
                "--trusted-host",
                "pypi.org",
                "--trusted-host",
                "files.pythonhosted.org",
            ])

        pip_command.extend([
            "-e",
            ".[dev]",
        ])

        run(pip_command)
        print("\n[2/2] Running test suite with coverage...")
    else:
        print("Running test suite with existing environment; no packages will be installed.")

    run([
        sys.executable,
        "-m",
        "pytest",
        "--cov=pymkm",
        "--cov-report=term-missing",
        "tests",
    ])

    print("\nTests completed successfully.")


if __name__ == "__main__":
    main()
