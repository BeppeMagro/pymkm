import pytest
import pandas as pd
import numpy as np
import warnings
import pickle

from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.utils.geometry_tools import DEFAULT_BASE_POINTS
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


def test_save_stores_parameter_metadata(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    table.table["C"] = {
        "data": pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]}),
        "params": {},
        "stopping_power_info": {},
    }

    path = tmp_path / "metadata.pkl"
    table.save(path)

    with path.open("rb") as f:
        payload = pickle.load(f)

    assert payload["__pymkm_mktable__"] == 1
    assert payload["model_version"] == "classic"
    assert payload["parameters"]["domain_radius"] == pytest.approx(0.3)
    assert payload["parameters"]["beta0"] == pytest.approx(0.05)
    assert "C" in payload["table"]


def test_load_rejects_parameter_mismatch_without_replacing_table(tmp_path):
    stored_params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    stored_table = MKTable(parameters=stored_params)
    stored_table.table["C"] = {
        "data": pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]}),
        "params": {},
        "stopping_power_info": {},
    }
    path = tmp_path / "mismatch.pkl"
    stored_table.save(path)

    current = MKTable(
        parameters=MKTableParameters(domain_radius=0.4, nucleus_radius=5.0, beta0=0.05)
    )
    sentinel = {"sentinel": {}}
    current.table = sentinel

    with pytest.raises(ValueError, match="domain_radius"):
        current.load(path)

    assert current.table is sentinel


def test_load_rejects_mcf_alpha0_mismatch(tmp_path):
    stored = MKTable(
        parameters=MKTableParameters(
            domain_radius=0.26,
            nucleus_radius=4.0,
            alpha0=0.15,
            beta0=0.05,
            use_mcf_model=True,
        )
    )
    stored.table["C"] = {
        "data": pd.DataFrame({"energy": [100.0], "c_bar": [0.9], "z_bar_c": [1.2]}),
        "params": {},
        "stopping_power_info": {},
    }
    path = tmp_path / "mcf.pkl"
    stored.save(path)

    current = MKTable(
        parameters=MKTableParameters(
            domain_radius=0.26,
            nucleus_radius=4.0,
            alpha0=0.16,
            beta0=0.05,
            use_mcf_model=True,
        )
    )

    with pytest.raises(ValueError, match="alpha0"):
        current.load(path)


def test_load_legacy_pickle_warns_and_remains_supported(tmp_path):
    legacy_table = {
        "C": {
            "data": pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]}),
            "params": {},
            "stopping_power_info": {},
        }
    }
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as f:
        pickle.dump(legacy_table, f)

    table = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )

    with pytest.warns(UserWarning, match="legacy MKTable pickle"):
        table.load(path)

    assert "C" in table.table


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        ({"__pymkm_mktable__": 99}, "Unsupported MKTable pickle format version"),
        ({"__pymkm_mktable__": 1, "model_version": "classic"}, "missing fields"),
        ([], "Invalid MKTable pickle payload"),
    ],
)
def test_load_rejects_invalid_pickle_payloads(tmp_path, payload, expected_message):
    path = tmp_path / "invalid.pkl"
    with path.open("wb") as f:
        pickle.dump(payload, f)

    table = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )

    with pytest.raises(ValueError, match=expected_message):
        table.load(path)


def test_load_rejects_model_mismatch(tmp_path):
    path = tmp_path / "wrong_model.pkl"
    payload = {
        "__pymkm_mktable__": 1,
        "model_version": "stochastic",
        "parameters": {
            "domain_radius": 0.3,
            "nucleus_radius": 5.0,
            "z0": None,
            "alpha0": None,
            "beta0": 0.05,
        },
        "table": {},
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)

    table = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )

    with pytest.raises(ValueError, match="model mismatch"):
        table.load(path)



def test_load_accepts_derived_z0_from_computed_table(tmp_path):
    from pymkm.physics.specific_energy import SpecificEnergy

    stored = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )
    stored.params.z0 = round(
        SpecificEnergy.compute_saturation_parameter(
            domain_radius=stored.params.domain_radius,
            nucleus_radius=stored.params.nucleus_radius,
            beta0=stored.params.beta0,
        ),
        2,
    )
    stored.table["C"] = {
        "data": pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.1]}),
        "params": {},
        "stopping_power_info": {},
    }
    path = tmp_path / "derived_z0.pkl"
    stored.save(path)

    current = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )
    assert current.params.z0 is None

    current.load(path)

    assert "C" in current.table
    assert current.params.z0 is None


def test_load_rejects_non_dict_parameter_metadata(tmp_path):
    path = tmp_path / "invalid_params_type.pkl"
    payload = {
        "__pymkm_mktable__": 1,
        "model_version": "classic",
        "parameters": [],
        "table": {},
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)

    table = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )

    with pytest.raises(ValueError, match="Invalid MKTable parameter metadata"):
        table.load(path)


def test_load_rejects_non_dict_table_data(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    path = tmp_path / "invalid_table.pkl"
    payload = {
        "__pymkm_mktable__": 1,
        "model_version": "classic",
        "parameters": params.__dict__.copy(),
        "table": [],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)

    table = MKTable(parameters=params)

    with pytest.raises(ValueError, match="Invalid MKTable table data"):
        table.load(path)


def test_load_rejects_invalid_parameter_metadata(tmp_path):
    path = tmp_path / "invalid_params.pkl"
    payload = {
        "__pymkm_mktable__": 1,
        "model_version": "classic",
        "parameters": {"unexpected": 1},
        "table": {},
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)

    table = MKTable(
        parameters=MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    )

    with pytest.raises(ValueError, match="Invalid MKTable parameter metadata"):
        table.load(path)


# --- _default_filename ---
@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_default_filename_creates_valid_windows_safe_path():
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        beta0=0.05,
        z0=1.0,
        use_stochastic_model=True,
    )
    table = MKTable(parameters=params)
    path = table._default_filename(".pkl")

    assert path.suffix == ".pkl"
    assert path.parent.exists()
    assert "rdrd" not in path.name
    assert "Rnrn" not in path.name
    assert "z0z0" not in path.name
    assert "b0b0" not in path.name
    assert "_rd0.30_Rn5.0_z01.0_b00.0500_" in path.name
    assert not set('<>:"/\\|?*').intersection(path.name)
    assert "default-fluka_2020_0" in path.name


def test_default_filename_sanitizes_custom_source_info():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    table.sp_table_set.source_info = r"loaded:C:\data folder\custom/source?.json"

    path = table._default_filename("txt")

    assert path.suffix == ".txt"
    assert "loaded-C--data_folder-custom-source-.json" in path.name
    assert not set('<>:"/\\|?*').intersection(path.name)


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


# --- MCF-MKM parameter validation ---
def test_mktable_mcf_parameters_and_model_version():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
    )
    table = MKTable(parameters=params)

    assert table.model_version == "mcf"
    assert table.params.mcf_nucleus_mode == "scaled"


def test_mktable_mcf_integrated_nucleus_mode():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
        mcf_nucleus_mode="integrated",
    )
    table = MKTable(parameters=params)

    assert table.model_version == "mcf"
    assert table.params.mcf_nucleus_mode == "integrated"


def test_mktable_mcf_requires_alpha0():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        beta0=0.04,
        use_mcf_model=True,
    )
    with pytest.raises(ValueError, match="alpha0 is required"):
        MKTable(parameters=params)


def test_mktable_mcf_requires_beta0():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        use_mcf_model=True,
    )
    with pytest.raises(ValueError, match="beta0 is required"):
        MKTable(parameters=params)


def test_mktable_mcf_rejects_stochastic_model():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
        use_stochastic_model=True,
    )
    with pytest.raises(ValueError, match="incompatible with use_stochastic_model"):
        MKTable(parameters=params)


def test_mktable_mcf_rejects_oxygen_effect():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
        apply_oxygen_effect=True,
    )
    with pytest.raises(ValueError, match="not currently supported for MCF-MKM"):
        MKTable(parameters=params)


def test_mktable_mcf_rejects_invalid_nucleus_mode():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
        mcf_nucleus_mode="invalid",
    )
    with pytest.raises(ValueError, match="must be either 'scaled' or 'integrated'"):
        MKTable(parameters=params)


def test_mktable_mcf_warns_if_z0_is_provided():
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        z0=1.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
    )
    with pytest.warns(UserWarning, match="z0 is not used for MCF-MKM"):
        table = MKTable(parameters=params)

    assert table.params.z0 == 1.0


def test_mktable_model_version_backward_compatibility():
    classic = MKTable(
        parameters=MKTableParameters(
            domain_radius=0.3,
            nucleus_radius=5.0,
            beta0=0.05,
        )
    )
    stochastic = MKTable(
        parameters=MKTableParameters(
            domain_radius=0.3,
            nucleus_radius=5.0,
            z0=1.0,
            use_stochastic_model=True,
        )
    )

    assert classic.model_version == "classic"
    assert stochastic.model_version == "stochastic"


def test_mktable_mcf_summary_contains_alpha0_and_nucleus_mode(capsys):
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
        mcf_nucleus_mode="scaled",
    )
    table = MKTable(parameters=params)

    table.summary(verbose=True)
    out = capsys.readouterr().out

    assert "mcf" in out
    assert "α₀" in out
    assert "MCF nucleus mode" in out
    assert "scaled" in out


def _make_mcf_table_for_write_txt(*, nucleus_mode="scaled"):
    params = MKTableParameters(
        domain_radius=0.26,
        nucleus_radius=4.0,
        alpha0=0.15,
        beta0=0.04,
        use_mcf_model=True,
        mcf_nucleus_mode=nucleus_mode,
    )
    table = MKTable(parameters=params)
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6, "ion_symbol": "C"},
        "params": {},
        "data": pd.DataFrame({
            "energy": [100.0],
            "let": [0.01],
            "c_bar": [0.9],
            "z_bar_c": [1.2],
        }),
    }
    return table


def test_mktable_mcf_write_txt(tmp_path):
    table = _make_mcf_table_for_write_txt(nucleus_mode="scaled")
    path = tmp_path / "mcf.txt"

    table.write_txt(
        params={
            "CellType": "V79",
            "Alpha_ref": 0.20,
            "Beta_ref": 0.04,
            "Alpha0": 0.15,
            "Beta0": 0.04,
        },
        filename=path,
    )

    content = path.read_text()
    assert "CellType  V79" in content
    assert "Parameter Alpha_ref 0.2000" in content
    assert "Parameter Beta_ref 0.0400" in content
    assert "Parameter Alpha0 0.1500" in content
    assert "Parameter Beta0 0.0400" in content
    assert "Parameter DomainRadius 0.2600" in content
    assert "Parameter NucleusRadius 4.0000" in content
    assert "MCFNucleusMode" not in content
    assert "Fragment C" in content
    assert "1.00000e+02 9.00000e-01 1.20000e+00" in content


def test_mktable_mcf_write_txt_minimal_params(tmp_path):
    table = _make_mcf_table_for_write_txt(nucleus_mode="integrated")
    path = tmp_path / "mcf_minimal.txt"

    table.write_txt(params={"CellType": "V79"}, filename=path)

    content = path.read_text()
    assert "Parameter Alpha_ref" not in content
    assert "Parameter Beta_ref" not in content
    assert "MCFNucleusMode" not in content


def test_mktable_mcf_write_txt_model_mismatch(tmp_path):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    table.table["C"] = {
        "stopping_power_info": {"atomic_number": 6, "ion_symbol": "C"},
        "params": {},
        "data": pd.DataFrame({"energy": [1.0], "z_bar_star_domain": [0.2]}),
    }

    with pytest.raises(ValueError, match="MCF output requested"):
        table.write_txt(
            params={"CellType": "Test"},
            filename=tmp_path / "bad_mcf.txt",
            model="mcf",
        )


def test_mktable_mcf_write_txt_alpha0_mismatch(tmp_path):
    table = _make_mcf_table_for_write_txt()
    with pytest.raises(ValueError, match="Mismatch between Alpha0"):
        table.write_txt(
            params={"CellType": "V79", "Alpha0": 0.16},
            filename=tmp_path / "bad_alpha.txt",
        )


def test_mktable_mcf_write_txt_beta0_mismatch(tmp_path):
    table = _make_mcf_table_for_write_txt()
    with pytest.raises(ValueError, match="Mismatch between Beta0"):
        table.write_txt(
            params={"CellType": "V79", "Beta0": 0.05},
            filename=tmp_path / "bad_beta.txt",
        )


@pytest.mark.parametrize("missing_column", ["c_bar", "z_bar_c"])
def test_mktable_mcf_write_txt_missing_column(tmp_path, missing_column):
    table = _make_mcf_table_for_write_txt()
    table.table["C"]["data"] = table.table["C"]["data"].drop(columns=[missing_column])

    with pytest.raises(KeyError, match=f"Missing expected column '{missing_column}'"):
        table.write_txt(
            params={"CellType": "V79"},
            filename=tmp_path / f"missing_{missing_column}.txt",
        )

def test_write_txt_rejects_unknown_model(tmp_path):
    table = _make_mcf_table_for_write_txt()

    with pytest.raises(
        ValueError,
        match=r"Unsupported model 'unsupported'.*classic.*mcf.*stochastic",
    ):
        table.write_txt(
            params={"CellType": "V79"},
            filename=tmp_path / "invalid_model.txt",
            model="unsupported",
        )



def test_mktable_parameters_default_base_points_use_shared_constant():
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0)
    assert params.base_points_b == DEFAULT_BASE_POINTS
    assert params.base_points_r == DEFAULT_BASE_POINTS
