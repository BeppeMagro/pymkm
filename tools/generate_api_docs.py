"""Regenerate the Sphinx API reference from the pymkm package structure."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "pymkm"
DOCS_SOURCE_DIR = ROOT / "docs" / "source"


def main() -> None:
    # Remove previously generated API stubs so deleted/renamed modules
    # cannot leave stale documentation behind.
    old_stubs = sorted(DOCS_SOURCE_DIR.glob("pymkm*.rst"))
    for path in old_stubs:
        path.unlink()

    print(f"Removed {len(old_stubs)} existing API stub(s).")

    command = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-f",  # overwrite generated files
        "-e",  # one page per module
        "-T",  # do not generate modules.rst
        "-M",  # document package/module before submodules
        "-o",
        str(DOCS_SOURCE_DIR),
        str(PACKAGE_DIR),
    ]

    print("\n>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)

    # The package root is only an API navigation page. Avoid documenting
    # pymkm.__init__ members again, which would duplicate indexed objects.
    root_stub = DOCS_SOURCE_DIR / "pymkm.rst"
    if root_stub.exists():
        text = root_stub.read_text(encoding="utf-8")
        text = text.replace(
            "   :members:\n"
            "   :undoc-members:\n"
            "   :show-inheritance:\n",
            "   :noindex:\n"
            "   :no-members:\n",
            1,
        )
        with root_stub.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)

    # Normalize generated RST files to LF line endings.
    generated = sorted(DOCS_SOURCE_DIR.glob("pymkm*.rst"))
    for path in generated:
        text = path.read_text(encoding="utf-8")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text.replace("\r\n", "\n"))

    print(f"\nGenerated {len(generated)} API stub(s) in {DOCS_SOURCE_DIR}")


if __name__ == "__main__":
    main()
