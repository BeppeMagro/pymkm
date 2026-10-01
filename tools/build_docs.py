"""Regenerate and build the pyMKM Sphinx documentation."""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"
BUILD_DIR = DOCS_DIR / "build"
HTML_DIR = BUILD_DIR / "html"
API_GENERATOR = ROOT / "tools" / "generate_api_docs.py"


def run(command: list[str]) -> None:
    print("\n>", " ".join(str(item) for item in command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    print("[1/3] Regenerating API documentation...")
    run([sys.executable, str(API_GENERATOR)])

    print("\n[2/3] Cleaning previous documentation build...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    print("\n[3/3] Building HTML documentation...")
    run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(SOURCE_DIR),
            str(HTML_DIR),
        ]
    )

    print(f"\nDocumentation built successfully:\n{HTML_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
