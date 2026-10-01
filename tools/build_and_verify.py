from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(command):
    print("\n>", " ".join(str(x) for x in command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    if DIST.exists():
        shutil.rmtree(DIST)

    print("[1/3] Building source and wheel distributions...")
    run([sys.executable, "-m", "build"])

    artifacts = sorted(DIST.glob("*"))
    if not artifacts:
        raise RuntimeError("No distribution artifacts were generated.")

    print("\n[2/3] Verifying distributions with twine...")
    run([sys.executable, "-m", "twine", "check", *artifacts])

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
