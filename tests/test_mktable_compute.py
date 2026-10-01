import pytest
import numpy as np
import pandas as pd
from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.mktable.compute import _compute_for_energy_let_pair, _run_energy_let_task, _build_worker_params
from pymkm.io.stopping_power import StoppingPowerTable


def create_dummy_table(ion_input="C"):
    energy = np.logspace(1, 3, 150)
    let = 1e-3 * energy**(-0.5)
    energy = energy[:5]  # Use only 5 energy points to reduce computation time  # Limit for speed
    let = let[:5]
    table = StoppingPowerTable(
        ion_input=ion_input,
        energy=energy,
        let=let,
        mass_number=12,
        source_program="mstar_3_12", # bypasses the >=150 point validation check
        ionization_potential=10.0
    )
    table.color = "blue"
    table.target = "WATER_LIQUID"
    return table

def test__compute_for_energy_let_pair_basic():
    params = dict(
        model_name="Kiefer-Chatterjee",
        core_radius_type="constant",
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=0.85,
        base_points_b=50,
        base_points_r=50,
        use_stochastic_model=False,
        integration_method="trapz"
    )
    result = _compute_for_energy_let_pair(params, energy=100.0, let=0.01, atomic_number=6)
    assert "z_bar_star_domain" in result
    assert isinstance(result["z_bar_star_domain"], float)

def test__compute_for_energy_let_pair_stochastic():
    params = dict(
        model_name="Kiefer-Chatterjee",
        core_radius_type="constant",
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=0.85,
        base_points_b=30,
        base_points_r=30,
        use_stochastic_model=True,
        integration_method="trapz"
    )
    result = _compute_for_energy_let_pair(params, energy=100.0, let=0.01, atomic_number=6)
    assert all(k in result for k in ["z_bar_star_domain", "z_bar_domain", "z_bar_nucleus"])

def test__compute_for_ion_serial(monkeypatch):
    # Patch expensive SpecificEnergy methods to speed up test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(5), np.linspace(0, 1, 5)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, base_points_b=10, base_points_r=10)
    table = MKTable(parameters=params)  # Initialize with reduced base points for testing
    ion = "Carbon"
    dummy = create_dummy_table(ion)
    table.sp_table_set.add(ion, dummy)

    ion_name, results = table._compute_for_ion(ion, parallel=False)
    assert ion_name == ion
    assert isinstance(results, list)
    assert "z_bar_star_domain" in results[0]

def test_compute_full(monkeypatch):
    # Patch expensive SpecificEnergy methods to speed up test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(5), np.linspace(0, 1, 5)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    dummy = create_dummy_table("Carbon")
    table.sp_table_set.add("Carbon", dummy)

    table.compute(parallel=False)
    assert "Carbon" in table.table
    df = table.table["Carbon"]["data"]
    assert isinstance(df, pd.DataFrame)
    assert "z_bar_star_domain" in df.columns

def test_compute_z0_fallback(monkeypatch):
    # Patch expensive SpecificEnergy methods to speed up test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(5), np.linspace(0, 1, 5)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, z0=None)
    table = MKTable(parameters=params)
    dummy = create_dummy_table("Carbon")
    table.sp_table_set.add("Carbon", dummy)

    assert table.params.z0 is None
    table.compute(parallel=False)
    assert table.params.z0 is not None

def test_run_energy_let_task_executes():
    def mock_func(x, y): return x + y
    result = _run_energy_let_task(mock_func, (2, 3))
    assert result == 5

def test__compute_for_ion_parallel_equivalent(monkeypatch):
    # NOTE: This test runs in serial mode to avoid multiprocessing issues during test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    monkeypatch.setattr("pymkm.utils.parallel.optimal_worker_count", lambda jobs: 1)

    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, base_points_b=5, base_points_r=5)
    table = MKTable(parameters=params)
    table.sp_table_set.add("Carbon", create_dummy_table("Carbon"))

    ion_name, results = table._compute_for_ion("Carbon", parallel=False)
    assert ion_name == "Carbon"
    assert isinstance(results, list)


def test_compute_with_custom_energy(monkeypatch):
    # Patch SpecificEnergy to avoid full computation
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)

    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    dummy = create_dummy_table("Carbon")
    table.sp_table_set.add("Carbon", dummy)

    custom_energy = [50.0, 100.0]
    table.compute(energy=custom_energy, parallel=False)

    assert "Carbon" in table.table


def test__compute_for_ion_parallel_flag_coverage(monkeypatch):
    # Simulate the parallel=True block using a fake executor to avoid true multiprocessing
    class FakeExecutor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, func, jobs): return [func(job) for job in jobs]

    monkeypatch.setattr("pymkm.mktable.compute.ProcessPoolExecutor", lambda *a, **kw: FakeExecutor())
    monkeypatch.setattr("pymkm.mktable.compute.tqdm", lambda *a, **kw: iter([]))
    monkeypatch.setattr("pymkm.utils.parallel.optimal_worker_count", lambda jobs: 1)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)

    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, base_points_b=5, base_points_r=5)
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    # This covers the parallel=True branch structurally using fake executor
    table._compute_for_ion("C", parallel=True)


def test_summary_verbose_outputs(capsys):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    table.sp_table_set.add("Carbon", create_dummy_table("Carbon"))

    table.summary(verbose=True)
    out = capsys.readouterr().out

    assert "MKTable Configuration" in out
    assert "Model version" in out
    assert "Stopping power source" in out
    assert "Carbon" in out
    assert "Track structure model" in out


def test_compute_runtime_error():
    table = MKTable.__new__(MKTable)  # Directly allocate MKTable without calling __init__  # bypass init
    table.sp_table_set = None
    table.params = None

    with pytest.raises(RuntimeError, match="MKTable is not properly initialized"):
        compute = getattr(MKTable, "compute")
        compute(table)

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_compute_for_ion_with_oxygen_effect(capsys):
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=1.0,
        beta0=0.05,
        use_stochastic_model=True,
        apply_oxygen_effect=True,
        pO2=5.0,
        K=3.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=2.0
    )
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    ion, result = table._compute_for_ion("C", parallel=False)

    captured = capsys.readouterr()
    assert "✔ Using OSMK2023-corrected values:" in captured.out
    assert ion == "C"
    assert isinstance(result, list)

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_compute_for_ion_without_oxygen_effect(capsys):
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=1.0,
        beta0=0.05,
        use_stochastic_model=True,
        apply_oxygen_effect=False  # No oxygen effect
    )
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    ion, result = table._compute_for_ion("C", parallel=False)

    captured = capsys.readouterr()
    assert "✔ Using OSMK2023-corrected values:" not in captured.out
    assert ion == "C"
    assert isinstance(result, list)

def test__compute_for_ion_respects_number_of_workers(monkeypatch):
    # This test verifies that the number_of_workers argument is correctly passed to the parallel executor in _compute_for_ion.
    called_workers = {} # Track the number of workers passed to FakeExecutor

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            called_workers['count'] = kwargs.get('max_workers', None)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, func, jobs): return [func(job) for job in jobs]

    monkeypatch.setattr("pymkm.mktable.compute.ProcessPoolExecutor", lambda *a, **kw: FakeExecutor(*a, **kw))
    monkeypatch.setattr("pymkm.mktable.compute.tqdm", lambda *a, **kw: iter([]))
    monkeypatch.setattr("pymkm.utils.parallel.optimal_worker_count", lambda jobs: 1)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)

    params = MKTableParameters(domain_radius=0.3,
                               nucleus_radius=5.0,
                               beta0=0.05, 
                               base_points_b=5, 
                               base_points_r=5)
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    table._compute_for_ion("C", parallel=True, number_of_workers=1)

    assert called_workers['count'] == 1



# --- MCF-MKM computation ---
def _mcf_worker_params(nucleus_mode="scaled"):
    return dict(
        model_name="Kiefer-Chatterjee",
        core_radius_type="constant",
        domain_radius=0.5,
        nucleus_radius=5.0,
        z0=None,
        alpha0=0.2,
        beta0=0.05,
        base_points_b=5,
        base_points_r=5,
        use_stochastic_model=False,
        use_mcf_model=True,
        mcf_nucleus_mode=nucleus_mode,
        integration_method="trapz",
    )


def test__compute_for_energy_let_pair_mcf_scaled(monkeypatch):
    b = np.linspace(0.0, 3.0, 4)
    z_domain = np.array([4.0, 3.0, 2.0, 1.0])

    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        lambda self, **kwargs: (z_domain.copy(), b.copy()),
    )

    result = _compute_for_energy_let_pair(
        _mcf_worker_params("scaled"),
        energy=100.0,
        let=0.01,
        atomic_number=6,
    )

    assert set(result) == {"c_bar", "z_bar_c"}
    assert np.isfinite(result["c_bar"])
    assert np.isfinite(result["z_bar_c"])
    assert 0.0 < result["c_bar"] <= 1.0
    assert result["z_bar_c"] > 0.0


def test__compute_for_energy_let_pair_mcf_integrated_uses_domain_b_grid(monkeypatch):
    b = np.linspace(0.0, 3.0, 4)
    z_domain = np.array([4.0, 3.0, 2.0, 1.0])
    z_nucleus = np.array([0.4, 0.3, 0.2, 0.1])
    calls = []

    def fake_single_event(self, **kwargs):
        calls.append((self.region_radius, kwargs.get("impact_parameters")))
        if np.isclose(self.region_radius, 0.5):
            return z_domain.copy(), b.copy()
        return z_nucleus.copy(), np.asarray(kwargs["impact_parameters"]).copy()

    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        fake_single_event,
    )

    result = _compute_for_energy_let_pair(
        _mcf_worker_params("integrated"),
        energy=100.0,
        let=0.01,
        atomic_number=6,
    )

    assert set(result) == {"c_bar", "z_bar_c"}
    assert len(calls) == 2
    assert calls[0][1] is None
    assert np.allclose(calls[1][1], b)


def test_mcf_scaled_and_integrated_routes_are_distinct(monkeypatch):
    b = np.linspace(0.0, 3.0, 4)
    z_domain = np.array([4.0, 3.0, 2.0, 1.0])
    z_nucleus_integrated = np.array([0.8, 0.6, 0.4, 0.2])

    def fake_single_event(self, **kwargs):
        if np.isclose(self.region_radius, 0.5):
            return z_domain.copy(), b.copy()
        return z_nucleus_integrated.copy(), np.asarray(kwargs["impact_parameters"]).copy()

    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        fake_single_event,
    )

    scaled = _compute_for_energy_let_pair(
        _mcf_worker_params("scaled"), 100.0, 0.01, 6
    )
    integrated = _compute_for_energy_let_pair(
        _mcf_worker_params("integrated"), 100.0, 0.01, 6
    )

    assert not np.isclose(scaled["c_bar"], integrated["c_bar"])
    assert not np.isclose(scaled["z_bar_c"], integrated["z_bar_c"])


def test_compute_full_mcf_table_contains_only_mcf_quantities(monkeypatch):
    b = np.linspace(0.0, 3.0, 4)
    z_domain = np.array([4.0, 3.0, 2.0, 1.0])

    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        lambda self, **kwargs: (z_domain.copy(), b.copy()),
    )

    params = MKTableParameters(
        domain_radius=0.5,
        nucleus_radius=5.0,
        alpha0=0.2,
        beta0=0.05,
        use_mcf_model=True,
        mcf_nucleus_mode="scaled",
        base_points_b=5,
        base_points_r=5,
    )
    table = MKTable(parameters=params)
    table.sp_table_set.add("Carbon", create_dummy_table("Carbon"))

    assert table.params.z0 is None
    table.compute(ions=["Carbon"], parallel=False)
    assert table.params.z0 is None

    df = table.table["Carbon"]["data"]
    assert list(df.columns) == ["energy", "let", "c_bar", "z_bar_c"]
    assert np.all(np.isfinite(df["c_bar"]))
    assert np.all(np.isfinite(df["z_bar_c"]))


def test__compute_for_energy_let_pair_mcf_invalid_nucleus_mode(monkeypatch):
    b = np.linspace(0.0, 3.0, 4)
    z_domain = np.array([4.0, 3.0, 2.0, 1.0])

    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        lambda self, **kwargs: (z_domain.copy(), b.copy()),
    )

    params = _mcf_worker_params("invalid")

    with pytest.raises(
        ValueError,
        match="mcf_nucleus_mode must be either 'scaled' or 'integrated'",
    ):
        _compute_for_energy_let_pair(
            params,
            energy=100.0,
            let=0.01,
            atomic_number=6,
        )

def test__build_worker_params_uses_mktable_values():
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        alpha0=0.12,
        beta0=0.05,
        base_points_b=12,
        base_points_r=34,
    )
    table = MKTable(parameters=params)

    worker_params = _build_worker_params(table, integration_method="simps")

    assert worker_params == {
        "model_name": params.model_name,
        "core_radius_type": params.core_radius_type,
        "base_points_b": 12,
        "base_points_r": 34,
        "domain_radius": 0.3,
        "nucleus_radius": 5.0,
        "z0": None,
        "alpha0": 0.12,
        "beta0": 0.05,
        "use_stochastic_model": False,
        "use_mcf_model": False,
        "mcf_nucleus_mode": "scaled",
        "integration_method": "simps",
    }


def test__build_worker_params_applies_effective_overrides():
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        beta0=0.05,
    )
    table = MKTable(parameters=params)

    worker_params = _build_worker_params(
        table,
        domain_radius=0.24,
        z0=1.10,
    )

    assert worker_params["domain_radius"] == 0.24
    assert worker_params["z0"] == 1.10
    assert worker_params["integration_method"] == "trapz"

