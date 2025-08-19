import os
import pytest
import tempfile
from pathlib import Path
from pymkm.io.data_registry import (
    get_default_txt_path,
    get_available_sources,
    list_available_defaults,
    load_lookup_table
)

# Utility to simulate a fake "spec" with an origin.
class FakeSpec:
    def __init__(self, origin):
        self.origin = origin

# ------------------------------
# Tests for get_default_txt_path

def test_get_default_txt_path_installed(tmp_path, monkeypatch):
    # Simulate the "installed" branch:
    fake_origin = tmp_path / "dummy_origin.txt"
    fake_origin.write_text("dummy")
    expected_file = tmp_path / "dummy.txt"
    expected_file.write_text("content")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: FakeSpec(str(fake_origin)))
    result = get_default_txt_path("any_source", "dummy.txt")
    assert result == str(expected_file)

def test_get_default_txt_path_fallback(monkeypatch):
    # Force fallback branch by making find_spec return None.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    source = "fallback_source"
    filename = "fallback.txt"
    # The fallback path is computed relative to the location of pymkm/io/data_registry.py.
    fallback_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pymkm", "data", "defaults", source, filename)
    )
    # Force os.path.exists to always return True.
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    result = get_default_txt_path(source, filename)
    # Compare as Path objects to normalize differences.
    assert Path(result) == Path(fallback_path)

def test_get_default_txt_path_exception(monkeypatch):
    # Force an exception when calling find_spec, which should trigger the except branch.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: (_ for _ in ()).throw(Exception("Forced exception")))
    # Ensure that os.path.exists returns False so that fallback is not found.
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    with pytest.raises(FileNotFoundError, match="Cannot find file 'file.txt' for source 'source'"):
        get_default_txt_path("source", "file.txt")

# ------------------------------
# Tests for get_available_sources

def test_get_available_sources_files(monkeypatch):
    class FakeDir:
        def __init__(self, name):
            self.name = name
        def is_dir(self):
            return True
    class FakeFiles:
        def iterdir(self):
            return [FakeDir("source1"), FakeDir("source2")]
    monkeypatch.setattr("importlib.resources.files", lambda pkg: FakeFiles())
    sources = get_available_sources()
    assert "source1" in sources and "source2" in sources

def test_get_available_sources_fallback(monkeypatch):
    def fake_files(pkg):
        raise Exception("fail")
    monkeypatch.setattr("importlib.resources.files", fake_files)
    temp_dir = tempfile.TemporaryDirectory()
    sourceA_dir = os.path.join(temp_dir.name, "sourceA")
    os.makedirs(sourceA_dir, exist_ok=True)
    fake_init = os.path.join(sourceA_dir, "__init__.py")
    with open(fake_init, "w") as f:
        f.write("")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: FakeSpec(fake_init))
    subdir = os.path.join(sourceA_dir, "subsource")
    os.makedirs(subdir, exist_ok=True)
    sources = get_available_sources()
    assert "subsource" in sources
    temp_dir.cleanup()

def test_get_available_sources_not_found(monkeypatch):
    monkeypatch.setattr("importlib.resources.files", lambda pkg: (_ for _ in ()).throw(Exception("fail")))
    # Force fallback branch to fail by returning a FakeSpec with origin None.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: FakeSpec(None))
    with pytest.raises(FileNotFoundError, match="Could not locate default sources."):
        get_available_sources()

# ------------------------------
# Tests for list_available_defaults

def test_list_available_defaults_files(monkeypatch):
    class FakeFile:
        def __init__(self, name, suffix):
            self.name = name
            self.suffix = suffix
    class FakeFolder:
        def iterdir(self):
            return [FakeFile("a.txt", ".txt"), FakeFile("b.txt", ".txt"), FakeFile("note.doc", ".doc")]
    monkeypatch.setattr("importlib.resources.files", lambda pkg: FakeFolder())
    defaults = list_available_defaults("any_source")
    assert "a.txt" in defaults
    assert "b.txt" in defaults
    assert "note.doc" not in defaults

def test_list_available_defaults_fallback(monkeypatch, tmp_path):
    def fake_files(pkg):
        raise Exception("fail")
    monkeypatch.setattr("importlib.resources.files", fake_files)
    source = "test_source"
    defaults_dir = tmp_path / "defaults" / source
    defaults_dir.mkdir(parents=True)
    file1 = defaults_dir / "file1.txt"
    file2 = defaults_dir / "file2.txt"
    file1.write_text("dummy")
    file2.write_text("dummy")
    (defaults_dir / "__init__.py").write_text("")
    fake_spec_origin = str(defaults_dir / "__init__.py")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: FakeSpec(fake_spec_origin))
    defaults = list_available_defaults(source)
    assert "file1.txt" in defaults
    assert "file2.txt" in defaults

def test_list_available_defaults_not_found(monkeypatch):
    def fake_files(pkg):
        raise Exception("fail")
    monkeypatch.setattr("importlib.resources.files", fake_files)
    # Force fallback branch failure: find_spec returns FakeSpec with origin None.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: FakeSpec(None))
    with pytest.raises(FileNotFoundError, match="Could not locate txt files for source: missing_source"):
        list_available_defaults("missing_source")

# ------------------------------
# Tests for load_lookup_table

def test_load_lookup_table_files(monkeypatch, tmp_path):
    json_content = '{"Carbon": {"atomic_number": 6, "mass_number": 12}}'
    elements_file = tmp_path / "elements.json"
    elements_file.write_text(json_content)
    class FakeFiles:
        def joinpath(self, subpath):
            return str(elements_file)
    monkeypatch.setattr("importlib.resources.files", lambda pkg: FakeFiles())
    table = load_lookup_table()
    assert "Carbon" in table
    assert table["Carbon"]["atomic_number"] == 6

def test_load_lookup_table_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("importlib.resources.files", lambda pkg: (_ for _ in ()).throw(Exception("fail")))
    json_content = '{"Carbon": {"atomic_number": 6, "mass_number": 12}}'
    fallback_file = tmp_path / "elements.json"
    fallback_file.write_text(json_content)
    monkeypatch.setattr("pymkm.io.data_registry.os.path.abspath", lambda x: str(fallback_file))
    table = load_lookup_table()
    assert "Carbon" in table
    assert table["Carbon"]["atomic_number"] == 6

def test_load_lookup_table_invalid_json(monkeypatch, tmp_path):
    invalid_content = "not a valid json"
    elements_file = tmp_path / "elements.json"
    elements_file.write_text(invalid_content)
    monkeypatch.setattr("pymkm.io.data_registry.importlib.resources.files", lambda pkg: (_ for _ in ()).throw(Exception("fail")))
    monkeypatch.setattr("pymkm.io.data_registry.os.path.abspath", lambda x: str(elements_file))
    with pytest.raises(ValueError):
        load_lookup_table()