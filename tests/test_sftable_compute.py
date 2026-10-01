import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.sftable.core import SFTable, SFTableParameters


def make_mock_table(stochastic=False, interpolate_multiple=False, mcf=False):
    if mcf:
        params = MKTableParameters(
            domain_radius=0.3,
            nucleus_radius=5.0,
            alpha0=0.1,
            beta0=0.05,
            use_mcf_model=True,
        )
    else:
        params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
        if stochastic:
            params.use_stochastic_model = True
    table = MKTable(parameters=params)

    # Mock required methods and attributes
    mock_sp = MagicMock()
    mock_sp.atomic_number = 6
    if interpolate_multiple:
        mock_sp.interpolate.side_effect = lambda energy=None, let=None: (
            [0.01] if energy is not None else {float(let): [100.0, 110.0]}
        )
    else:
        mock_sp.interpolate.side_effect = lambda energy=None, let=None: (
            [0.01] if energy is not None else {float(let): [100.0]}
        )
    table.sp_table_set.get = MagicMock(return_value=mock_sp)
    table.sp_table_set._map_to_fullname = lambda ion: "Carbon"
    return table

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_classic(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic")
    results = sft.table
    assert isinstance(results, list)
    assert len(results) == 1
    df = results[0]["data"]
    assert "dose" in df.columns and "survival_fraction" in df.columns

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_stochastic(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="stochastic")
    results = sft.table
    df = results[0]["data"]
    assert "survival_fraction" in df.columns

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_stochastic_model_mismatch_raises(mock_compute):
    table = make_mock_table(stochastic=False)
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="Stochastic output requested"):
        sft.compute(ion="C", energy=100.0, model="stochastic")

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_only_reuses_precomputed_row(mock_compute):
    table = make_mock_table()
    df = pd.DataFrame({
        "energy": [100.0],
        "let": [0.01],
        "z_bar_star_domain": [0.25],
    })
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table
    assert results[0]["calculation_info"] == "precomputed"
    mock_compute.assert_not_called()

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_only_interpolated_from_existing_table(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({
        "energy": [100.0, 100.0],
        "let": [0.01, 0.02],
        "z_bar_star_domain": [0.1, 0.2],
    })
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table   
    assert len(results) == 2
    assert all(r["params"]["energy"] == 100.0 for r in results)
    assert all(r["calculation_info"] == "precomputed" for r in results)
    mock_compute.assert_not_called()

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_let_only_interpolated_from_existing_table(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({
        "energy": [100.0],
        "let": [0.01],
        "z_bar_star_domain": [0.1],
    })
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", let=0.01, model="classic", force_recompute=False)
    results = sft.table  
    assert len(results) == 1
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "precomputed"
    mock_compute.assert_not_called()

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_and_let_both_present(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({
        "energy": [100.0],
        "let": [0.01],
        "z_bar_star_domain": [0.1],
    })
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, let=0.01, model="classic", force_recompute=False)
    results = sft.table
    assert results[0]["calculation_info"] == "precomputed"
    mock_compute.assert_not_called()

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_missing_energy_and_let_raises(mock_compute):
    table = make_mock_table()
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="At least one of 'energy' or 'let' must be specified"):
        sft.compute(ion="C")

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_let_only_multiple_energies(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table(interpolate_multiple=True)
    # Do not preload table.table so that ion_data is None and interpolation is triggered
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", let=0.01, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 2
    assert all(r["params"]["let"] == 0.01 for r in results)
    assert set(r["params"]["energy"] for r in results) == {100.0, 110.0}

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_only_infers_let(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    table.table = {}  # Ensure ion_data is None so it triggers interpolation
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 1
    assert results[0]["params"]["energy"] == 100.0
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "computed"

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_and_let_both_given_no_data(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    table.table = {}  # Force use of final else branch
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, let=0.01, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 1
    assert results[0]["params"]["energy"] == 100.0
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "computed"

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_not_found_triggers_interpolation(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({"energy": [90.0], "let": [0.03]})
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 1
    assert results[0]["params"]["energy"] == 100.0
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "computed"

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_osmk2023_path(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03,
        alphaS=0.07,
        beta0=0.05,
        pO2=5.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=3.0
    )
    sft = SFTable(parameters=params)
    original_domain_radius = table.params.domain_radius
    original_z0 = table.params.z0

    sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)
    results = sft.table

    assert results[0]["params"]["osmk_version"] == "2023"
    assert results[0]["calculation_info"] == "computed"
    assert "survival_fraction" in results[0]["data"].columns
    assert table.params.domain_radius == original_domain_radius
    assert table.params.z0 == original_z0
    assert mock_compute.call_count == 2

    base_worker_params = mock_compute.call_args_list[0].args[0]
    osmk_worker_params = mock_compute.call_args_list[1].args[0]
    assert base_worker_params["domain_radius"] == pytest.approx(original_domain_radius)
    assert osmk_worker_params["domain_radius"] != pytest.approx(original_domain_radius)
    assert osmk_worker_params["z0"] != base_worker_params["z0"]

def test_osmk_rejected_for_classic_model():
    table = make_mock_table(stochastic=False)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0, f_rd_max=1.2, f_z0_max=1.8, Rmax=3.0
    )
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="Oxygen effect.*only be applied with model='stochastic'"):
        sft.compute(ion="C", energy=100.0, model="classic", apply_oxygen_effect=True)

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_osmk_2021_path(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1, "z_bar_domain": 0.05, "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0, zR=1.5, gamma=2.0, Rm=1.2
    )
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)
    assert sft.table[0]["params"]["osmk_version"] == "2021"

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
def test_osmk_raises_if_both_versions_specified():
    table = make_mock_table(stochastic=True)
    with pytest.raises(ValueError, match="Cannot mix OSMK 2021.*2023"):
        SFTableParameters(
            mktable=table,
            alphaL=0.03, alphaS=0.07, beta0=0.05,
            pO2=5.0,
            zR=1.5, gamma=2.0, Rm=1.2,      # OSMK 2021
            f_rd_max=1.5, f_z0_max=2.0, Rmax=3.0  # OSMK 2023
        )

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
def test_osmk_raises_if_missing_all_parameters():
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0  # No zR/gamma/Rm nor f_rd_max/f_z0_max/Rmax
    )
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="required parameters are missing"):
        sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_osmk_inconsistent_versions_detected_in_compute(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1, "z_bar_domain": 0.05, "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)

    # Costruisci con SOLO parametri 2021
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0,
        zR=1.5, gamma=2.0, Rm=1.2
    )
    # Assegna *dopo* i parametri 2023 (forzando inconsistenza in compute)
    params.f_rd_max = 1.5
    params.f_z0_max = 2.0
    params.Rmax = 3.0

    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="cannot provide both 2021 and 2023"):
        sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_mcf(mock_compute):
    mock_compute.return_value = {
        "c_bar": 0.8,
        "z_bar_c": 0.3,
    }
    table = make_mock_table(mcf=True)
    dose_grid = np.array([0.0, 1.0, 2.0])
    params = SFTableParameters(
        mktable=table,
        alpha0=0.1,
        beta0=0.05,
        dose_grid=dose_grid,
    )
    sft = SFTable(parameters=params)

    with patch(
        "pymkm.sftable.compute.SpecificEnergy.compute_saturation_parameter"
    ) as mock_saturation:
        sft.compute(ion="C", energy=100.0)

    alpha_mcf = 0.1 * 0.8 + 0.05 * 0.3
    beta_mcf = 0.05 * 0.8 ** 2
    expected = np.exp(-alpha_mcf * dose_grid - beta_mcf * dose_grid ** 2)

    result = sft.table[0]
    assert result["params"]["model"] == "mcf"
    np.testing.assert_allclose(result["data"]["survival_fraction"].to_numpy(), expected)
    mock_saturation.assert_not_called()

    worker_params = mock_compute.call_args.args[0]
    assert worker_params["use_mcf_model"] is True
    assert worker_params["alpha0"] == pytest.approx(0.1)
    assert worker_params["beta0"] == pytest.approx(0.05)
    assert worker_params["mcf_nucleus_mode"] == "scaled"


@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_mcf_integrated_mode_is_forwarded(mock_compute):
    mock_compute.return_value = {
        "c_bar": 0.9,
        "z_bar_c": 0.2,
    }
    table = make_mock_table(mcf=True)
    table.params.mcf_nucleus_mode = "integrated"
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    sft.compute(ion="C", energy=100.0, model="mcf")

    worker_params = mock_compute.call_args.args[0]
    assert worker_params["mcf_nucleus_mode"] == "integrated"


def test_compute_mcf_model_mismatch_raises():
    table = make_mock_table()
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    with pytest.raises(ValueError, match="MCF-MKM output requested"):
        sft.compute(ion="C", energy=100.0, model="mcf")


def test_compute_mcf_table_rejects_non_mcf_model():
    table = make_mock_table(mcf=True)
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    with pytest.raises(ValueError, match="MCF-mode MKTable can only be used"):
        sft.compute(ion="C", energy=100.0, model="classic")


def test_compute_invalid_model_raises():
    table = make_mock_table()
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    with pytest.raises(ValueError, match="model must be one of"):
        sft.compute(ion="C", energy=100.0, model="invalid")

@patch("pymkm.sftable.compute.compute_mcf_lq_coefficients")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_mcf_uses_biology_model(
    mock_compute,
    mock_mcf_lq,
):
    mock_compute.return_value = {
        "c_bar": 0.8,
        "z_bar_c": 0.3,
    }
    mock_mcf_lq.return_value = (0.095, 0.032)

    table = make_mock_table(mcf=True)
    dose_grid = np.array([0.0, 1.0])
    params = SFTableParameters(
        mktable=table,
        alpha0=0.1,
        beta0=0.05,
        dose_grid=dose_grid,
    )
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0)

    mock_mcf_lq.assert_called_once_with(
        c_bar=0.8,
        z_bar_c=0.3,
        alpha0=0.1,
        beta0=0.05,
    )
    expected = np.exp(-0.095 * dose_grid - 0.032 * dose_grid ** 2)
    np.testing.assert_allclose(
        sft.table[0]["data"]["survival_fraction"].to_numpy(),
        expected,
    )

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute.compute_smk_gamma")
@patch("pymkm.sftable.compute.compute_smk_lq_coefficients")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_stochastic_uses_biology_gamma(
    mock_compute,
    mock_smk_lq,
    mock_smk_gamma,
):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2,
    }
    mock_smk_lq.return_value = (0.15, 0.04)

    dose_grid = np.array([0.0, 1.0, 2.0])
    gamma = np.array([0.01, 0.02, 0.03])
    mock_smk_gamma.return_value = gamma

    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alpha0=0.1,
        beta0=0.05,
        dose_grid=dose_grid,
    )
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="stochastic")

    mock_smk_gamma.assert_called_once_with(
        alpha_smk=0.15,
        beta_smk=0.04,
        z_bar_nucleus=0.2,
        dose=dose_grid,
    )

    expected = (
        np.exp(-0.15 * dose_grid - 0.04 * dose_grid ** 2)
        * (1 + gamma * dose_grid)
    )
    np.testing.assert_allclose(
        sft.table[0]["data"]["survival_fraction"].to_numpy(),
        expected,
    )

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_precomputed_row_missing_required_quantity_falls_back_to_compute(mock_compute):
    mock_compute.return_value = {"z_bar_star_domain": 0.1}
    table = make_mock_table()
    table.table = {
        "Carbon": {
            "data": pd.DataFrame({"energy": [100.0], "let": [0.01]})
        }
    }
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    sft.compute(
        ion="C",
        energy=100.0,
        let=0.01,
        model="classic",
        force_recompute=False,
    )

    assert sft.table[0]["calculation_info"] == "computed"
    mock_compute.assert_called_once()


@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_mcf_reuses_precomputed_quantities(mock_compute):
    table = make_mock_table(mcf=True)
    table.table = {
        "Carbon": {
            "data": pd.DataFrame({
                "energy": [100.0],
                "let": [0.01],
                "c_bar": [0.8],
                "z_bar_c": [0.3],
            })
        }
    }
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    sft.compute(
        ion="C",
        energy=100.0,
        let=0.01,
        model="mcf",
        force_recompute=False,
    )

    assert sft.table[0]["calculation_info"] == "precomputed"
    mock_compute.assert_not_called()


@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_osmk2023_reuses_base_table_without_mutating_it(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.12,
        "z_bar_domain": 0.06,
        "z_bar_nucleus": 0.21,
    }
    table = make_mock_table(stochastic=True)
    table.table = {
        "Carbon": {
            "data": pd.DataFrame({
                "energy": [100.0],
                "let": [0.01],
                "z_bar_star_domain": [0.1],
                "z_bar_domain": [0.05],
                "z_bar_nucleus": [0.2],
            })
        }
    }
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03,
        alphaS=0.07,
        beta0=0.05,
        pO2=5.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=3.0,
    )
    sft = SFTable(parameters=params)
    original_domain_radius = table.params.domain_radius
    original_z0 = table.params.z0

    sft.compute(
        ion="C",
        energy=100.0,
        model="stochastic",
        apply_oxygen_effect=True,
        force_recompute=False,
    )

    # The base stochastic quantities are reused; only the OSMK-2023 effective
    # geometry is recomputed.
    assert mock_compute.call_count == 1
    assert sft.table[0]["calculation_info"] == "computed"
    assert table.params.domain_radius == original_domain_radius
    assert table.params.z0 == original_z0


@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_let_not_found_in_precomputed_table_falls_back_to_interpolation(mock_compute):
    mock_compute.return_value = {"z_bar_star_domain": 0.1}
    table = make_mock_table(interpolate_multiple=True)
    table.table = {
        "Carbon": {
            "data": pd.DataFrame({
                "energy": [90.0],
                "let": [0.03],
                "z_bar_star_domain": [0.2],
            })
        }
    }
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    sft.compute(ion="C", let=0.01, model="classic", force_recompute=False)

    assert len(sft.table) == 2
    assert all(result["calculation_info"] == "computed" for result in sft.table)
    assert mock_compute.call_count == 2
