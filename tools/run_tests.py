from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("\n>", " ".join(str(x) for x in command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    print("[1/2] Installing pyMKM in editable mode with dev dependencies...")
    run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    print("\n[2/2] Running test suite with coverage...")
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
