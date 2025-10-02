import pytest
import pandas as pd
import numpy as np
import warnings

from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.io.stopping_power import StoppingPowerTable


# Automatically redirect Path.home() to tmp_path to avoid polluting ~/.pyMKM
@pytest.fixture(autouse=True)
def redirect_home_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


# Utility to create a dummy stopping power table for ion C
def create_dummy_table(symbol="C"):
    energy = np.array([100.0])
    let = np.array([0.01])
    table = StoppingPowerTable(
        ion_input=symbol,
        energy=energy,
        let=let,
        mass_number=12,
        source_program="dummy",
        ionization_potential=10.0
    )
    table.color = "blue"
    table.target = "WATER_LIQUID"
    return table


# --- MKTableParameters ---
def test_mktable_parameters_repr_and_dict():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    assert isinstance(repr(params), str)
    assert hasattr(params, '__dict__')


def test_mktable_parameters_from_dict():
    d = {"domain_radius": 0.3, "nucleus_radius": 5.0, "beta0": 0.05}
    params = MKTableParameters.from_dict(d)
    assert isinstance(params, MKTableParameters)

def test_mktable_parameters_from_dict_raises_on_extra_keys():
    bad_dict = {
        "domain_radius": 0.3,
        "nucleus_radius": 5.0,
        "beta0": 0.05,
        "unexpected_key": 42
    }
    with pytest.raises(ValueError, match="Unrecognized keys"):
        MKTableParameters.from_dict(bad_dict)


# --- MKTable construction and validation ---
def test_mktable_requires_beta_or_z0():
    with pytest.raises(ValueError, match="Both z0 and beta0 are missing"):
        MKTable(MKTableParameters(domain_radius=0.3, nucleus_radius=5.0))


def test_mktable_repr_and_display(monkeypatch, capsys):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    assert isinstance(repr(table), str)

    # display without data should raise
    with pytest.raises(ValueError, match="No computed results found"):
        table.display()

    # simulate a filled table
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["C"] = {"data": df, "params": {}, "stopping_power_info": {}}
    table.display()
    out = capsys.readouterr().out
    assert "z_bar_star_domain" in out

def test_summary_verbose_with_ions(capsys):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    # Inject a fake ion into sp_table_set
    table.sp_table_set.get_available_ions = lambda: ["C", "He"]
    table.summary(verbose=True)
    out = capsys.readouterr().out
    assert "Available ions" in out
    assert "Track structure model" in out
    
def test_refresh_parameters_detects_change(capsys):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    # Simulate a change in domain_radius
    old_params = MKTableParameters(domain_radius=0.1, nucleus_radius=5.0, beta0=0.05)
    table._refresh_parameters(original_params=old_params)
    out = capsys.readouterr().out
    assert "MKTableParameters updated" in out
    assert "domain_radius" in out

def test_display_full_dataframe(capsys):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": np.linspace(1.0, 10.0, 15),
        "z_bar_star_domain": np.random.rand(15)
    })
    table.table["C"] = {
        "stopping_power_info": {"source": "mock", "atomic_number": 6},
        "params": {"some_param": 1},
        "data": df
    }
    table.display(preview_rows=5)
    out = capsys.readouterr().out
    assert "Top 5 rows" in out
    assert "Bottom 5 rows" in out
    assert "some_param" in out

def test_mktable_warns_if_z0_missing_for_smk():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, use_stochastic_model=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MKTable(parameters=params)
        assert any("z0 not provided" in str(warning.message) for warning in w)

def test_mktable_warns_if_z0_provided_in_mkm_without_beta0():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, z0=1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MKTable(parameters=params)
        assert any("z0 provided but beta0 is missing" in str(warning.message) for warning in w)

def test_get_table_returns_dataframe():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    
    # Simulate a computed table
    df = pd.DataFrame({"energy": [1.0], "let": [0.02], "z_bar_star_domain": [0.1]})
    table.table["Carbon"] = {"data": df, "params": {}, "stopping_power_info": {"atomic_number": 6}}

    # Test get_table using ion name
    out_df = table.get_table("Carbon")
    assert isinstance(out_df, pd.DataFrame)
    assert "z_bar_star_domain" in out_df.columns

    # Test get_table using atomic number
    out_df_zn = table.get_table(6)
    assert isinstance(out_df_zn, pd.DataFrame)
    assert np.allclose(out_df["z_bar_star_domain"], out_df_zn["z_bar_star_domain"])

def test_get_table_raises_if_not_computed():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)

    with pytest.raises(ValueError, match="No computed results found"):
        table.get_table("C")

def test_get_table_raises_if_ion_not_found():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["Carbon"] = {"data": df, "params": {}, "stopping_power_info": {"atomic_number": 6}}

    with pytest.raises(ValueError, match="Ion 'O' not found"):
        table.get_table("O")

    
# --- save/load ---
def test_save_and_load_roundtrip(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["C"] = {"data": df, "params": {}, "stopping_power_info": {}}

    path = tmp_path / "test.pkl"
    table.save(path)
    assert path.exists()

    new_table = MKTable(parameters=params)
    new_table.load(path)
    assert "C" in new_table.table

def test_save_raises_without_table(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    with pytest.raises(ValueError, match="Cannot save"):
        table.save(tmp_path / "out.pkl")

def test_load_raises_if_file_missing(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    missing_path = tmp_path / "nonexistent.pkl"
    with pytest.raises(FileNotFoundError, match="File not found"):
        table.load(missing_path)


# --- _default_filename ---
@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_default_filename_creates_valid_path():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, z0=1.0, use_stochastic_model=True)
    table = MKTable(parameters=params)
    path = table._default_filename(".pkl")
    assert path.suffix == ".pkl"
    assert path.parent.exists()


# --- write_txt ---
@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_write_txt_defaults_to_all_ions(tmp_path):
    """Ensure that if max_atomic_number is None, all available ions are included."""
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    # Simula due ioni con Z diversi
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.2]})
    table.table["H"] = {
        "stopping_power_info": {"atomic_number": 1, "ion_symbol": "H"},
        "params": {},
        "data": df
    }
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6, "ion_symbol": "C"},
        "params": {},
        "data": df
    }
    path = tmp_path / "default_all.txt"
    # non passo max_atomic_number → deve includere H e C
    table.write_txt(
        params={"CellType": "HSG", "Alpha_0": 0.1, "Beta": 0.05},
        filename=path,
        model="classic"
    )
    content = path.read_text()
    assert "Fragment H" in content
    assert "Fragment C" in content

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_write_txt_smk_beta_both_provided(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.06, z0=1.0, use_stochastic_model=True)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": [1.0],
        "z_bar_domain": [0.1],
        "z_bar_star_domain": [0.2],
        "z_bar_nucleus": [0.3]
    })
    table.table["Carbon"] = {
        "stopping_power_info": {"atomic_number": 6, "source": "mock", "target": "water", "ion_symbol": "C"},
        "params": {},
        "data": df
    }
    path = tmp_path / "smk_full.txt"
    table.write_txt(
        params={
            "CellType": "T", "Alpha_ref": 0.1, "Beta_ref": 0.05,
            "scale_factor": 1.0,
            "Alpha0": 0.12,
            "Beta0": 0.06
        },
        filename=path,
        model="stochastic",
        max_atomic_number=6
    )
    assert path.exists()
    assert "Beta0 0.060" in path.read_text()
    assert "Fragment C" in path.read_text()


def test_write_txt_smk_beta_only_in_dict(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, z0=1.0, use_stochastic_model=True)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": [1.0],
        "z_bar_domain": [0.1],
        "z_bar_star_domain": [0.2],
        "z_bar_nucleus": [0.3]
    })
    table.table["Carbon"] = {
        "stopping_power_info": {"atomic_number": 6, "source": "mock", "target": "water"},
        "params": {},
        "data": df
    }
    path = tmp_path / "smk_dict.txt"
    table.write_txt(
        params={
            "CellType": "T", "Alpha_ref": 0.1, "Beta_ref": 0.05,
            "scale_factor": 1.0,
            "Alpha0": 0.12,
            "Beta0": 0.07
        },
        filename=path,
        model="stochastic",
        max_atomic_number=6
    )
    assert "Beta0 0.070" in path.read_text()


def test_write_txt_stochastic_model_mismatch():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.2]})
    table.table["Carbon"] = {
        "stopping_power_info": {"atomic_number": 6, "source": "mock", "target": "water"},
        "params": {},
        "data": df
    }
    with pytest.raises(ValueError, match="Stochastic output requested"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_ref": 0.1, "Beta_ref": 0.05, "Alpha0": 0.12},
            model="stochastic",
            max_atomic_number=6
        )

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
@pytest.mark.filterwarnings("ignore:'scale_factor' not provided.*")
def test_write_txt_scale_factor_warning(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, z0=1.0, beta0=0.05, use_stochastic_model=True)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": [1.0],
        "z_bar_domain": [0.1],
        "z_bar_star_domain": [0.2],
        "z_bar_nucleus": [0.3]
    })
    table.table["Carbon"] = {
        "stopping_power_info": {"atomic_number": 6, "source": "mock", "target": "water", "ion_symbol": "C"},
        "params": {},
        "data": df
    }
    path = tmp_path / "smk.txt"
    table.write_txt(
        params={"CellType": "T", "Alpha_ref": 0.1, "Beta_ref": 0.05, "Alpha0": 0.1, "Beta0": 0.05},
        filename=path,
        model="stochastic",
        max_atomic_number=6
    )
    assert path.exists()
    assert "Fragment C" in path.read_text()

def test_write_txt_classic(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.06)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": [1.0],
        "z_bar_star_domain": [0.2]
    })
    table.table["Carbon"] = {
        "stopping_power_info": {"atomic_number": 6, "source": "mock", "target": "water", "ion_symbol": "C"},
        "params": {},
        "data": df
    }
    path = tmp_path / "classic.txt"
    table.write_txt(
        params={"CellType": "HSG", "Alpha_0": 0.1, "Beta": 0.06},
        filename=path,
        model="classic",
        max_atomic_number=6
    )
    content = path.read_text()
    assert "Parameter Alpha_0 0.100" in content
    assert "Parameter Beta 0.060" in content
    assert "Fragment C" in content

def test_write_txt_raises_if_table_empty():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    with pytest.raises(ValueError, match="Cannot write: MKTable has not been computed yet"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_0": 0.1, "Beta": 0.05},
            model="classic",
            max_atomic_number=6
        )

def test_write_txt_raises_missing_required_keys():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.2]})
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6, "source": "mock", "target": "water"},
        "params": {},
        "data": df
    }

    with pytest.raises(KeyError, match="Missing required keys"):
        table.write_txt(
            params={"Alpha_0": 0.1},  # manca 'CellType'
            model="classic",
            max_atomic_number=6
        )

def test_write_txt_raises_on_unexpected_keys(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6},
        "params": {},
        "data": df
    }
    with pytest.raises(KeyError, match="Unexpected keys"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_0": 0.1, "Beta": 0.05, "ExtraKey": 1.0},
            model="classic",
            filename=tmp_path / "out.txt",
            max_atomic_number=6
        )


def test_write_txt_raises_if_requested_z_exceeds():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6},
        "params": {},
        "data": df
    }
    with pytest.raises(ValueError, match="exceeds computed table max Z"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_0": 0.1, "Beta": 0.05},
            model="classic",
            filename="dummy.txt",
            max_atomic_number=10  # > 6
        )


def test_write_txt_classic_beta_mismatch(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.06)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6},
        "params": {},
        "data": df
    }
    with pytest.raises(ValueError, match="Mismatch between beta0 in params"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_0": 0.1, "Beta": 0.05},  # diverso da beta0
            model="classic",
            filename=tmp_path / "mismatch.txt",
            max_atomic_number=6
        )

@pytest.mark.filterwarnings("ignore:z0 provided but beta0 is missing.*")
def test_write_txt_classic_beta_missing_everywhere(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=None, z0=1.0)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.2]})
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6},
        "params": {},
        "data": df
    }

    with pytest.raises(ValueError, match="Beta must be defined either in params"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_0": 0.1},  # manca 'Beta'
            model="classic",
            filename=tmp_path / "classic_missing_beta.txt",
            max_atomic_number=6
        )

@pytest.mark.filterwarnings("ignore:'scale_factor' not provided.*")
def test_write_txt_smk_beta0_missing_everywhere(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=None, z0=1.0, use_stochastic_model=True)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": [1.0],
        "z_bar_domain": [0.1],
        "z_bar_star_domain": [0.2],
        "z_bar_nucleus": [0.3]
    })
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6},
        "params": {},
        "data": df
    }
    with pytest.raises(ValueError, match="Beta0 must be defined"):
        table.write_txt(
            params={"CellType": "Test", "Alpha_ref": 0.1, "Beta_ref": 0.05, "Alpha0": 0.1},
            model="stochastic",
            filename=tmp_path / "smk_missing.txt",
            max_atomic_number=6
        )

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
@pytest.mark.filterwarnings("ignore:'scale_factor' not provided.*")
def test_write_txt_smk_beta0_mismatch(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.07, z0=1.0, use_stochastic_model=True)
    table = MKTable(parameters=params)
    df = pd.DataFrame({
        "energy": [1.0],
        "z_bar_domain": [0.1],
        "z_bar_star_domain": [0.2],
        "z_bar_nucleus": [0.3]
    })
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6},
        "params": {},
        "data": df
    }
    with pytest.raises(ValueError, match="Mismatch between Beta0 in params"):
        table.write_txt(
            params={
                "CellType": "Test", "Alpha_ref": 0.1, "Beta_ref": 0.05,
                "Alpha0": 0.1, "Beta0": 0.05
            },
            model="stochastic",
            filename=tmp_path / "smk_mismatch.txt",
            max_atomic_number=6
        )


def test_write_txt_skips_high_Z_only(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]})
    table.table["B"] = {"stopping_power_info": {"atomic_number": 10}, "params": {}, "data": df}
    table.write_txt(
        params={"CellType": "T", "Alpha_0": 0.1, "Beta": 0.05},
        model="classic",
        filename=tmp_path / "skip.txt",
        max_atomic_number=6
    )
    assert "Fragment" not in (tmp_path / "skip.txt").read_text()


def test_write_txt_raises_missing_column_classic(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0]})
    table.table["C"] = {"stopping_power_info": {"atomic_number": 6}, "params": {}, "data": df}
    with pytest.raises(KeyError, match="Missing expected column 'z_bar_star_domain'"):
        table.write_txt(
            params={"CellType": "T", "Alpha_0": 0.1, "Beta": 0.05},
            model="classic",
            filename=tmp_path / "error.txt",
            max_atomic_number=6
        )

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided. z0 will be used for SMK.*")
@pytest.mark.filterwarnings("ignore:'scale_factor' not provided.*")
def test_write_txt_raises_missing_column_stochastic(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, z0=1.0, use_stochastic_model=True)
    table = MKTable(parameters=params)
    df = pd.DataFrame({"energy": [1.0], "z_bar_domain": [0.1]})
    table.table["C"] = {"stopping_power_info": {"atomic_number": 6}, "params": {}, "data": df}
    with pytest.raises(KeyError, match="Missing expected column 'z_bar_star_domain'"):
        table.write_txt(
            params={"CellType": "T", "Alpha_ref": 0.1, "Beta_ref": 0.05, "Alpha0": 0.1, "Beta0": 0.05},
            model="stochastic",
            filename=tmp_path / "error_smk.txt",
            max_atomic_number=6
        )

def test_mktable_raises_if_oxygen_effect_without_stochastic():
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        beta0=0.05,
        apply_oxygen_effect=True,
        use_stochastic_model=False
    )
    with pytest.raises(ValueError, match="apply_oxygen_effect=True requires use_stochastic_model=True"):
        MKTable(parameters=params)

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
def test_mktable_raises_if_oxygen_effect_missing_params():
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        beta0=0.05,
        apply_oxygen_effect=True,
        use_stochastic_model=True,
        pO2=None,
        f_rd_max=1.2,
        f_z0_max=1.5,
        Rmax=None
    )
    with pytest.raises(ValueError, match="apply_oxygen_effect=True but missing OSMK 2023 parameters:"):
        MKTable(parameters=params)
