import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace

from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.sftable.core import SFTable, SFTableParameters

# Setup mock MKTable
params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
mock_table = MKTable(parameters=params)

# --- SFTableParameters tests ---
def test_sftableparameters_valid_init():
    p = SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05)
    assert p.beta0 == 0.05
    assert isinstance(p.dose_grid, np.ndarray)

def test_sftableparameters_accepts_list_as_dose_grid():
    p = SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05, dose_grid=[0, 1, 2])
    assert isinstance(p.dose_grid, np.ndarray)
    assert np.allclose(p.dose_grid, np.array([0, 1, 2]))

def test_sftableparameters_beta0_required_if_not_in_table():
    valid_params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=valid_params)
    table.params.beta0 = None
    with pytest.raises(ValueError, match="beta0 must be provided either explicitly or via MKTable.params"):
        SFTableParameters(mktable=table, alpha0=0.1)

def test_sftableparameters_beta0_fallback_warns():
    params_no_beta = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.07)
    mock_table_local = MKTable(parameters=params_no_beta)
    with pytest.warns(UserWarning, match="beta0 not provided"):
        SFTableParameters(mktable=mock_table_local, alpha0=0.1)

def test_sftableparameters_beta0_mismatch_raises():
    with pytest.raises(ValueError, match="Mismatch between provided beta0"):
        SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.02)

def test_sftableparameters_invalid_mktable_type():
    with pytest.raises(TypeError):
        SFTableParameters(mktable="not_a_table", alpha0=0.1, beta0=0.05)

def test_sftableparameters_from_dict_success():
    config = {"mktable": mock_table, "alpha0": 0.1, "beta0": 0.05}
    p = SFTableParameters.from_dict(config)
    assert isinstance(p, SFTableParameters)

def test_sftableparameters_from_dict_invalid_key():
    with pytest.raises(ValueError, match="Unrecognized keys"):
        SFTableParameters.from_dict({"mktable": mock_table, "alpha0": 0.1, "beta0": 0.05, "extra": 1})


@pytest.mark.parametrize("use_stochastic_model", [False, True])
def test_sftableparameters_alpha0_required_for_normoxic_mkm_models(use_stochastic_model):
    if use_stochastic_model:
        params_local = MKTableParameters(
            domain_radius=0.3,
            nucleus_radius=5.0,
            z0=50.0,
            use_stochastic_model=True,
        )
        table = MKTable(parameters=params_local)
        table.params.beta0 = 0.05
    else:
        table = mock_table

    with pytest.raises(
        ValueError,
        match="alpha0 must be provided for normoxic MKM/SMK survival calculations",
    ):
        SFTableParameters(mktable=table, beta0=0.05)


# --- SFTable tests ---
def test_sftable_repr():
    sft = SFTable(SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05))
    assert "<SFTable" in repr(sft)

def test_sftable_display_empty_raises():
    sft = SFTable(SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05))
    with pytest.raises(ValueError, match="No results to display"):
        sft.display([])

def test_sftable_display_full_output(capsys):
    sft = SFTable(SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05))
    df = pd.DataFrame({"Dose [Gy]": [0, 2, 4], "SF": [1.0, 0.8, 0.5]})
    results = [{
        "params": {"cell_line": "HSG"},
        "calculation_info": "test run",
        "data": df
    }]
    sft.display(results)
    output = capsys.readouterr().out
    assert "Survival Fraction Results" in output
    assert "cell_line" in output
    assert "Dose [Gy]" in output

def test_sftable_display_with_empty_dataframe(capsys):
    sft = SFTable(SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05))
    results = [{
        "params": {"cell_line": "HSG"},
        "calculation_info": "test run",
        "data": pd.DataFrame()
    }]
    sft.display(results)
    output = capsys.readouterr().out
    assert "No data found in this result." in output

def test_sftable_summary_output(capsys):
    sft = SFTable(SFTableParameters(mktable=mock_table, alpha0=0.1, beta0=0.05))
    sft.summary()
    output = capsys.readouterr().out
    assert "SFTable Configuration" in output
    assert "α_0" in output
    assert "β_0" in output

def test_osmk2023_alpha0_autocomputed():
    p = SFTableParameters(
        mktable=mock_table,
        alphaL=0.03,
        alphaS=0.07,
        beta0=0.05,
        pO2=5.0
    )
    assert p.alpha0 == 0.1

def test_osmk2023_alphaL_autocomputed():
    p = SFTableParameters(
        mktable=mock_table,
        alpha0=0.1,
        alphaS=0.07,
        beta0=0.05,
        pO2=5.0
    )
    assert np.isclose(p.alphaL, 0.03)

def test_osmk2023_alpha_values_missing():
    with pytest.raises(ValueError, match="at least two of alpha0, alphaL, and alphaS"):
        SFTableParameters(
            mktable=mock_table,
            alpha0=0.1,
            beta0=0.05,
            pO2=5.0
        )

def test_osmk_mixed_versions_error():
    with pytest.raises(ValueError, match="Cannot mix OSMK 2021.*2023"):
        SFTableParameters(
            mktable=mock_table,
            alpha0=0.1,
            alphaL=0.03,
            alphaS=0.07,
            beta0=0.05,
            pO2=5.0,
            zR=1.5,  # 2021
            f_rd_max=1.3  # 2023
        )

def test_summary_includes_osmk2023(capsys):
    sft = SFTable(SFTableParameters(
        mktable=mock_table,
        alpha0=0.1,
        alphaL=0.03,
        alphaS=0.07,
        beta0=0.05,
        pO2=5.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=3.0
    ))
    sft.summary()
    output = capsys.readouterr().out
    assert "f_rd_max" in output
    assert "f_z0_max" in output
    assert "Rmax" in output

def test_osmk2023_computes_alphaS_from_alpha0_alphaL():
    p = SFTableParameters(
        mktable=mock_table,
        alpha0=0.1,
        alphaL=0.03,
        beta0=0.05,
        pO2=5.0
    )
    assert np.isclose(p.alphaS, 0.07)

def test_osmk2023_inconsistent_alpha0_raises():
    with pytest.raises(ValueError, match=r"Inconsistent values for OSMK: alpha0=0.1, alphaL \+ alphaS = 0.11"):
        SFTableParameters(
            mktable=mock_table,
            alpha0=0.1,
            alphaL=0.05,
            alphaS=0.06,
            beta0=0.05,
            pO2=5.0
        )

def test_osmk2023_computes_alpha0_from_alphaL_alphaS():
    p = SFTableParameters(
        mktable=mock_table,
        alphaL=0.04,
        alphaS=0.06,
        beta0=0.05,
        pO2=5.0
    )
    assert np.isclose(p.alpha0, 0.10)

def test_sftable_repr_with_alphaL_and_alphaS():
    sft = SFTable(SFTableParameters(
        mktable=mock_table,
        alphaL=0.04,
        alphaS=0.06,
        beta0=0.05,
        pO2=5.0
    ))
    repr_str = repr(sft)
    assert "α_0 = 0.1" in repr_str
    assert "β_0 = 0.05" in repr_str

# --- MCF-MKM SFTableParameters tests ---
def _make_mcf_table(alpha0=0.12, beta0=0.045):
    mcf_params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        alpha0=alpha0,
        beta0=beta0,
        use_mcf_model=True,
    )
    return MKTable(parameters=mcf_params)


def test_sftableparameters_mcf_alpha0_beta0_fallback_warns():
    mcf_table = _make_mcf_table(alpha0=0.12, beta0=0.045)
    with pytest.warns(UserWarning) as record:
        p = SFTableParameters(mktable=mcf_table)

    messages = [str(w.message) for w in record]
    assert any("beta0 not provided" in message for message in messages)
    assert any("alpha0 not provided" in message for message in messages)
    assert np.isclose(p.alpha0, 0.12)
    assert np.isclose(p.beta0, 0.045)


def test_sftableparameters_mcf_matching_alpha0_explicit():
    mcf_table = _make_mcf_table(alpha0=0.12, beta0=0.045)
    p = SFTableParameters(
        mktable=mcf_table,
        alpha0=0.12,
        beta0=0.045,
    )
    assert np.isclose(p.alpha0, 0.12)
    assert np.isclose(p.beta0, 0.045)


def test_sftableparameters_mcf_alpha0_mismatch_raises():
    mcf_table = _make_mcf_table(alpha0=0.12, beta0=0.045)
    with pytest.raises(ValueError, match="Mismatch between provided alpha0"):
        SFTableParameters(
            mktable=mcf_table,
            alpha0=0.11,
            beta0=0.045,
        )


def test_sftableparameters_mcf_beta0_mismatch_raises():
    mcf_table = _make_mcf_table(alpha0=0.12, beta0=0.045)
    with pytest.raises(ValueError, match="Mismatch between provided beta0"):
        SFTableParameters(
            mktable=mcf_table,
            alpha0=0.12,
            beta0=0.04,
        )


def test_sftableparameters_mcf_rejects_osmk_parameters():
    mcf_table = _make_mcf_table(alpha0=0.12, beta0=0.045)
    with pytest.raises(ValueError, match="not supported for MCF-MKM"):
        SFTableParameters(
            mktable=mcf_table,
            alpha0=0.12,
            beta0=0.045,
            pO2=5.0,
            alphaL=0.04,
            alphaS=0.08,
        )

def test_sftableparameters_mcf_alpha0_required_if_not_in_table():
    mcf_table = _make_mcf_table(alpha0=0.12, beta0=0.045)
    mcf_table.params.alpha0 = None
    with pytest.raises(
        ValueError,
        match="alpha0 must be provided either explicitly or via MKTable.params for MCF-MKM",
    ):
        SFTableParameters(
            mktable=mcf_table,
            beta0=0.045,
        )

