"""
Discovery and loading of default stopping power data files.

This module provides functions to:

- Locate `.txt` files for specific ions and sources (e.g.`mstar_3_12`, `geant4_11_3_0`, `fluka_2020_0`)
- List available sources and files
- Load the `elements.json` periodic table lookup used throughout pyMKM

All file paths are resolved using :mod:`importlib.resources`, making them portable
within installed packages or local development environments.
"""

import os
import json
import importlib.util
from typing import List, Dict
from pathlib import Path

REQUIRED_HEADER_KEYS = ["Ion", "AtomicNumber", "MassNumber", "SourceProgram", "IonizationPotential", "Target"]

def get_default_txt_path(source: str, filename: str) -> str:
    """
    Locate a default .txt file for a given source and filename.

    Tries first to resolve the file within the installed package.
    Falls back to a local relative path (for development use) if needed.

    :param source: The name of the data subdirectory (e.g., "ions").
    :type source: str
    :param filename: The name of the .txt file to locate.
    :type filename: str

    :returns: Absolute path to the located file.
    :rtype: str

    :raises FileNotFoundError: If the file cannot be found in either location.
    """
    try:
        # Attempt to locate installed package data
        spec = importlib.util.find_spec(f"pymkm.data.defaults.{source}")
        if spec is not None and spec.origin is not None:
            base_dir = os.path.dirname(spec.origin)
            full_path = os.path.join(base_dir, filename)
            if os.path.exists(full_path):
                return full_path
    except Exception:
        pass

    # Local fallback (e.g. during development)
    local = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "defaults", source, filename)
    )
    if os.path.exists(local):
        return local

    raise FileNotFoundError(f"Cannot find file '{filename}' for source '{source}'")


def get_available_sources() -> List[str]:
    """
    List all available data sources within the default data registry.

    Looks for subdirectories under `pymkm.data.defaults` corresponding to different source categories
    and checks if the directory contains .txt files with adequate header.

    :returns: List of source names (subdirectory names).
    :rtype: list[str]

    :raises FileNotFoundError: If the base directory cannot be located.
    """
    try:
        from importlib.resources import files
        base = files("pymkm.data.defaults")
        all_subdirectories = [f for f in base.iterdir() if f.is_dir()]
    except Exception:
        spec = importlib.util.find_spec("pymkm.data.defaults")
        if spec and spec.origin:
            folder_path = Path(spec.origin).parent
            all_subdirectories = [f for f in folder_path.iterdir() if f.is_dir()]
        raise FileNotFoundError("Could not locate default sources.")
        
    subdirectories = []
    for subdir in all_subdirectories:
        # Check presence of __init__.py file 
        init_path = subdir / "__init__.py"
        if not init_path.is_file():
            continue
        txt_files = [f for f in subdir.iterdir() if f.suffix == ".txt"]
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            header = {line.split('=')[0].strip(): line.split('=')[1].strip()
                      for line in lines if '=' in line}
            missing = [key for key in REQUIRED_HEADER_KEYS if key not in header]
            if missing:
                raise FileNotFoundError(
                    f"Source files with unexpected header format in default source folder {subdir.name}: {', '.join(missing)}.\nCould not locate default sources."
                )
        subdirectories.append(subdir.name)

    return subdirectories


def list_available_defaults(source: str) -> List[str]:
    """
    List all available .txt files under a given source folder.

    Looks within `pymkm.data.defaults.<source>` and returns matching files.

    :param source: The name of the data source folder.
    :type source: str

    :returns: List of filenames with .txt extension.
    :rtype: list[str]

    :raises FileNotFoundError: If the source folder or .txt files cannot be located.
    """
    try:
        from importlib.resources import files
        folder = files(f"pymkm.data.defaults.{source}")
        return [f.name for f in folder.iterdir() if f.suffix == ".txt"]
    except Exception:
        spec = importlib.util.find_spec(f"pymkm.data.defaults.{source}")
        if spec and spec.origin:
            folder_path = Path(spec.origin).parent
            return [f.name for f in folder_path.iterdir() if f.suffix == ".txt"]
    raise FileNotFoundError(f"Could not locate txt files for source: {source}")


def load_lookup_table() -> Dict[str, Dict]:
    """
    Load the chemical elements lookup table from a JSON file.

    Tries to load `elements.json` from the installed package, with a fallback
    to a local relative file during development.

    :returns: Dictionary mapping element symbols to their properties.
    :rtype: dict[str, dict]

    :raises FileNotFoundError: If the elements.json file cannot be found.
    :raises json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        from importlib.resources import files
        path = files("pymkm.data").joinpath("elements.json")
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # Fallback for local use
        local_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "elements.json")
        )
        with open(local_path, "r") as f:
            return json.load(f)